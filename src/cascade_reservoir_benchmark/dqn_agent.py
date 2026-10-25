"""Pure Double DQN components for the centralized cascade controller.

This module deliberately contains no reservoir physics, observation conversion,
action-grid construction, or reward shaping.  Those concerns belong to the DQN
adapter and runner; this module only consumes normalized numeric states, joint
action indices, environment-produced boolean masks, and already-scaled rewards.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


AGENT_STATE_SCHEMA_VERSION = 4
REPLAY_STATE_SCHEMA_VERSION = 2
ACTION_MASK_MODE = "environment_transition_reachability_and_monthly_limits_v2"
INVALID_ACTION_VALUE = -1.0e9
INDEXED_OUTPUT_MODE = "indexed_output"
ACTION_CONDITIONED_MODE = "action_conditioned"
Q_NETWORK_MODES = (INDEXED_OUTPUT_MODE, ACTION_CONDITIONED_MODE)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch CPU/CUDA random number generators."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _finite_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


class QNetwork(nn.Module):
    """Fully connected Q-network with configurable hidden dimensions."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: Sequence[int],
        action_count: int,
    ) -> None:
        super().__init__()
        self.input_dimension = _positive_int("input_dimension", input_dimension)
        self.action_count = _positive_int("action_count", action_count)
        if isinstance(hidden_dimensions, (str, bytes)):
            raise TypeError("hidden_dimensions must be a sequence of integers.")
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", dimension)
            for index, dimension in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("hidden_dimensions cannot be empty.")

        layers: list[nn.Module] = []
        previous_dimension = self.input_dimension
        for dimension in self.hidden_dimensions:
            layers.extend(
                [
                    nn.Linear(previous_dimension, dimension),
                    nn.LayerNorm(dimension),
                    nn.ReLU(),
                ]
            )
            previous_dimension = dimension
        layers.append(nn.Linear(previous_dimension, self.action_count))
        self.layers = nn.Sequential(*layers)
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class ActionConditionedQNetwork(nn.Module):
    """Efficient scalar Q(s, a) with shared continuous action embeddings."""

    def __init__(
        self,
        state_dimension: int,
        hidden_dimensions: Sequence[int],
        action_feature_dimension: int = 2,
    ) -> None:
        super().__init__()
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.action_feature_dimension = _positive_int(
            "action_feature_dimension", action_feature_dimension
        )
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", dimension)
            for index, dimension in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("hidden_dimensions cannot be empty.")
        state_layers: list[nn.Module] = []
        previous = self.state_dimension
        for dimension in self.hidden_dimensions:
            state_layers.extend(
                [nn.Linear(previous, dimension), nn.LayerNorm(dimension), nn.ReLU()]
            )
            previous = dimension
        self.state_encoder = nn.Sequential(*state_layers)
        embedding_dimension = min(128, self.hidden_dimensions[-1])
        self.embedding_dimension = embedding_dimension
        self.state_value = nn.Linear(self.hidden_dimensions[-1], 1)
        self.state_action_embedding = nn.Linear(
            self.hidden_dimensions[-1], embedding_dimension
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_feature_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.action_bias = nn.Sequential(
            nn.Linear(self.action_feature_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, 1),
        )
        self.apply(QNetwork._initialize_layer)

    def _state_terms(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.state_encoder(states)
        return self.state_value(encoded), self.state_action_embedding(encoded)

    def score_selected(
        self, states: torch.Tensor, action_features: torch.Tensor
    ) -> torch.Tensor:
        value, state_embedding = self._state_terms(states)
        action_embedding = self.action_encoder(action_features)
        interaction = (state_embedding * action_embedding).sum(dim=1, keepdim=True)
        interaction = interaction / float(self.embedding_dimension) ** 0.5
        return value + interaction + self.action_bias(action_features)

    def score_all(
        self,
        states: torch.Tensor,
        action_features: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        value, state_embedding = self._state_terms(states)
        pieces: list[torch.Tensor] = []
        for start in range(0, len(action_features), chunk_size):
            features = action_features[start : start + chunk_size]
            action_embedding = self.action_encoder(features)
            interaction = state_embedding @ action_embedding.transpose(0, 1)
            interaction = interaction / float(self.embedding_dimension) ** 0.5
            bias = self.action_bias(features).transpose(0, 1)
            pieces.append(value + interaction + bias)
        return torch.cat(pieces, dim=1)


class UniformReplayBuffer:
    """Fixed-capacity, preallocated, uniform experience replay buffer.

    ``bootstrap_terminal`` is true only for a genuine MDP terminal transition.
    An artificial training-window boundary is a truncation and must be stored as
    false so its next state remains part of the Double DQN bootstrap target.
    """

    def __init__(self, capacity: int, state_dimension: int, action_count: int) -> None:
        self.capacity = _positive_int("capacity", capacity)
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.action_count = _positive_int("action_count", action_count)
        self.action_mask_bytes = (self.action_count + 7) // 8
        self.states = np.zeros((self.capacity, self.state_dimension), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dimension), dtype=np.float32)
        self.bootstrap_terminals = np.zeros(self.capacity, dtype=np.bool_)
        self.next_action_masks_packed = np.zeros(
            (self.capacity, self.action_mask_bytes), dtype=np.uint8
        )
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        bootstrap_terminal: bool,
        next_action_mask: Sequence[bool] | np.ndarray,
    ) -> None:
        state_array = self._validated_state("state", state)
        next_state_array = self._validated_state("next_state", next_state)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer index.")
        action = int(action)
        if action < 0:
            raise ValueError("action must be nonnegative.")
        reward = _finite_float("reward", reward)
        if not isinstance(bootstrap_terminal, (bool, np.bool_)):
            raise TypeError("bootstrap_terminal must be boolean.")
        mask = self._validated_action_mask(next_action_mask)

        self.states[self.position] = state_array
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state_array
        self.bootstrap_terminals[self.position] = bool(bootstrap_terminal)
        self.next_action_masks_packed[self.position] = np.packbits(
            mask, bitorder="little"
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _validated_state(
        self,
        name: str,
        state: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(state, dtype=np.float32)
        if result.shape != (self.state_dimension,):
            raise ValueError(
                f"{name} must have shape ({self.state_dimension},), got {result.shape}."
            )
        if np.any(~np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values.")
        return result

    def _validated_action_mask(
        self, action_mask: Sequence[bool] | np.ndarray
    ) -> np.ndarray:
        result = np.asarray(action_mask)
        if result.shape != (self.action_count,) or result.dtype != np.bool_:
            raise ValueError(
                f"action mask must be boolean with shape ({self.action_count},)."
            )
        if not np.any(result):
            raise ValueError("action mask must contain at least one feasible action.")
        return result

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        batch_size = _positive_int("batch_size", batch_size)
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from replay size {self.size}."
            )
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        next_action_masks = np.unpackbits(
            self.next_action_masks_packed[indices],
            axis=1,
            count=self.action_count,
            bitorder="little",
        ).astype(np.bool_, copy=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "bootstrap_terminals": self.bootstrap_terminals[indices],
            "next_action_masks": next_action_masks,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return all ring-buffer state required for exact training resume."""
        return {
            "schema_version": REPLAY_STATE_SCHEMA_VERSION,
            "capacity": self.capacity,
            "state_dimension": self.state_dimension,
            "action_count": self.action_count,
            "action_mask_mode": ACTION_MASK_MODE,
            "position": self.position,
            "size": self.size,
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_states": self.next_states.copy(),
            "bootstrap_terminals": self.bootstrap_terminals.copy(),
            "next_action_masks_packed": self.next_action_masks_packed.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a replay state after validating its structure and content."""
        if int(state.get("schema_version", -1)) != REPLAY_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported replay-buffer state schema.")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("Replay-buffer capacity does not match the configured agent.")
        if int(state.get("state_dimension", -1)) != self.state_dimension:
            raise ValueError("Replay-buffer state dimension does not match the configured agent.")
        if int(state.get("action_count", -1)) != self.action_count:
            raise ValueError("Replay-buffer action count does not match the configured agent.")
        if state.get("action_mask_mode") != ACTION_MASK_MODE:
            raise ValueError("Replay buffer does not contain compatible feasible-action masks.")

        size = int(state.get("size", -1))
        position = int(state.get("position", -1))
        if not 0 <= size <= self.capacity:
            raise ValueError("Replay-buffer size is invalid.")
        if not 0 <= position < self.capacity:
            raise ValueError("Replay-buffer write position is invalid.")
        if size < self.capacity and position != size:
            raise ValueError("Partially filled replay buffer has an inconsistent write position.")

        arrays = {
            "states": np.asarray(state["states"], dtype=np.float32),
            "actions": np.asarray(state["actions"], dtype=np.int64),
            "rewards": np.asarray(state["rewards"], dtype=np.float32),
            "next_states": np.asarray(state["next_states"], dtype=np.float32),
            "bootstrap_terminals": np.asarray(
                state["bootstrap_terminals"], dtype=np.bool_
            ),
            "next_action_masks_packed": np.asarray(
                state["next_action_masks_packed"], dtype=np.uint8
            ),
        }
        expected_shapes = {
            "states": (self.capacity, self.state_dimension),
            "actions": (self.capacity,),
            "rewards": (self.capacity,),
            "next_states": (self.capacity, self.state_dimension),
            "bootstrap_terminals": (self.capacity,),
            "next_action_masks_packed": (
                self.capacity,
                self.action_mask_bytes,
            ),
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"Replay-buffer {name} has shape {array.shape}; "
                    f"expected {expected_shapes[name]}."
                )
        if size and (
            np.any(~np.isfinite(arrays["states"][:size]))
            or np.any(~np.isfinite(arrays["rewards"][:size]))
            or np.any(~np.isfinite(arrays["next_states"][:size]))
            or np.any(arrays["actions"][:size] < 0)
        ):
            raise ValueError("Replay-buffer active transitions contain invalid values.")

        np.copyto(self.states, arrays["states"])
        np.copyto(self.actions, arrays["actions"])
        np.copyto(self.rewards, arrays["rewards"])
        np.copyto(self.next_states, arrays["next_states"])
        np.copyto(self.bootstrap_terminals, arrays["bootstrap_terminals"])
        np.copyto(
            self.next_action_masks_packed,
            arrays["next_action_masks_packed"],
        )
        self.position = position
        self.size = size


class DoubleDQNAgent:
    """Centralized Double DQN over one externally defined joint action space."""

    def __init__(
        self,
        *,
        state_dimension: int,
        action_count: int,
        hidden_dimensions: Sequence[int],
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_schedule: str,
        epsilon_decay_steps: int,
        learning_starts: int,
        target_update_interval: int,
        replay_capacity: int,
        batch_size: int,
        max_grad_norm: float | None,
        seed: int,
        device: str | torch.device,
        q_network_mode: str = INDEXED_OUTPUT_MODE,
        action_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        action_score_chunk_size: int = 256,
    ) -> None:
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.action_count = _positive_int("action_count", action_count)
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", dimension)
            for index, dimension in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("hidden_dimensions cannot be empty.")
        self.learning_rate = _finite_float("learning_rate", learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        self.gamma = _finite_float("gamma", gamma)
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one.")
        self.epsilon_start = _finite_float("epsilon_start", epsilon_start)
        self.epsilon_min = _finite_float("epsilon_min", epsilon_min)
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= min <= start <= 1.")
        if epsilon_schedule != "linear":
            raise ValueError("DQN V1 supports only epsilon_schedule='linear'.")
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_decay_steps = _positive_int(
            "epsilon_decay_steps", epsilon_decay_steps
        )
        if isinstance(learning_starts, bool) or not isinstance(
            learning_starts, (int, np.integer)
        ):
            raise TypeError("learning_starts must be an integer.")
        self.learning_starts = int(learning_starts)
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative.")
        self.target_update_interval = _positive_int(
            "target_update_interval", target_update_interval
        )
        self.replay_capacity = _positive_int("replay_capacity", replay_capacity)
        self.batch_size = _positive_int("batch_size", batch_size)
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity.")
        if max_grad_norm is None:
            self.max_grad_norm = None
        else:
            self.max_grad_norm = _finite_float("max_grad_norm", max_grad_norm)
            if self.max_grad_norm <= 0.0:
                raise ValueError("max_grad_norm must be positive when enabled.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        self.device = torch.device(device)
        if q_network_mode not in Q_NETWORK_MODES:
            raise ValueError(f"q_network_mode must be one of {list(Q_NETWORK_MODES)}.")
        self.q_network_mode = str(q_network_mode)
        self.action_score_chunk_size = _positive_int(
            "action_score_chunk_size", action_score_chunk_size
        )
        if self.q_network_mode == ACTION_CONDITIONED_MODE:
            features = np.asarray(action_features, dtype=np.float32)
            if (
                features.shape != (self.action_count, 2)
                or np.any(~np.isfinite(features))
                or np.any(features < 0.0)
                or np.any(features > 1.0)
            ):
                raise ValueError(
                    "Action-conditioned Q requires finite [action_count, 2] "
                    "features in [0, 1]."
                )
            if len(np.unique(features, axis=0)) != self.action_count:
                raise ValueError("Action-conditioned Q features must be unique.")
            self.action_features = features.copy()
        else:
            if action_features is not None:
                raise ValueError("Indexed-output Q does not accept action_features.")
            self.action_features = None

        set_global_seed(self.seed)
        if self.q_network_mode == ACTION_CONDITIONED_MODE:
            self.online_network = ActionConditionedQNetwork(
                self.state_dimension, self.hidden_dimensions
            ).to(self.device)
            self.target_network = ActionConditionedQNetwork(
                self.state_dimension, self.hidden_dimensions
            ).to(self.device)
        else:
            self.online_network = QNetwork(
                self.state_dimension, self.hidden_dimensions, self.action_count
            ).to(self.device)
            self.target_network = QNetwork(
                self.state_dimension, self.hidden_dimensions, self.action_count
            ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay = UniformReplayBuffer(
            self.replay_capacity, self.state_dimension, self.action_count
        )
        self.epsilon = self.epsilon_start
        self.env_steps = 0
        self.learning_steps = 0

    def epsilon_for_environment_steps(self, environment_steps: int) -> float:
        """Return the configured linear epsilon value for an env-step count."""
        if isinstance(environment_steps, bool) or not isinstance(
            environment_steps, (int, np.integer)
        ):
            raise TypeError("environment_steps must be an integer.")
        steps = int(environment_steps)
        if steps < 0:
            raise ValueError("environment_steps must be nonnegative.")
        progress = np.clip(
            (steps - self.learning_starts) / self.epsilon_decay_steps,
            0.0,
            1.0,
        )
        return float(
            self.epsilon_start
            + progress * (self.epsilon_min - self.epsilon_start)
        )

    def select_action(
        self,
        state: Sequence[float] | np.ndarray,
        *,
        greedy: bool = False,
        action_mask: Sequence[bool] | np.ndarray,
    ) -> int:
        """Choose only among physics-feasible actions without renumbering them."""
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dimension,):
            raise ValueError(
                f"state must have shape ({self.state_dimension},), got {state_array.shape}."
            )
        if np.any(~np.isfinite(state_array)):
            raise ValueError("state must contain only finite values.")
        mask = np.asarray(action_mask)
        if mask.shape != (self.action_count,) or mask.dtype != np.bool_:
            raise ValueError(
                f"action_mask must be boolean with shape ({self.action_count},)."
            )
        feasible_actions = np.flatnonzero(mask)
        if len(feasible_actions) == 0:
            raise ValueError("action_mask must contain at least one feasible action.")
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(feasible_actions))

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state_array, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self._all_action_q(self.online_network, state_tensor)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            masked_q_values = q_values.masked_fill(
                ~mask_tensor.unsqueeze(0), INVALID_ACTION_VALUE
            )
            return int(masked_q_values.argmax(dim=1).item())

    def _all_action_q(
        self, network: nn.Module, states: torch.Tensor
    ) -> torch.Tensor:
        """Score every stable action index for each state in bounded chunks."""
        if states.ndim != 2 or states.shape[1] != self.state_dimension:
            raise ValueError("states must have shape [batch, state_dimension].")
        if self.q_network_mode == INDEXED_OUTPUT_MODE:
            return network(states)
        if self.action_features is None:
            raise RuntimeError("Action-conditioned Q features are missing.")
        features = torch.as_tensor(
            self.action_features, dtype=torch.float32, device=self.device
        )
        if not isinstance(network, ActionConditionedQNetwork):
            raise TypeError("Action-conditioned mode requires its matching network.")
        return network.score_all(
            states, features, chunk_size=self.action_score_chunk_size
        )

    def _selected_action_q(
        self,
        network: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Score only the replay action paired with each batch state."""
        if self.q_network_mode == INDEXED_OUTPUT_MODE:
            return network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        if self.action_features is None:
            raise RuntimeError("Action-conditioned Q features are missing.")
        features = torch.as_tensor(
            self.action_features, dtype=torch.float32, device=self.device
        )
        selected_features = features.index_select(0, actions)
        if not isinstance(network, ActionConditionedQNetwork):
            raise TypeError("Action-conditioned mode requires its matching network.")
        return network.score_selected(states, selected_features).squeeze(1)

    def store_transition(
        self,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        next_state: Sequence[float] | np.ndarray,
        *,
        bootstrap_terminal: bool,
        next_action_mask: Sequence[bool] | np.ndarray,
    ) -> None:
        """Store one transition and count one environment interaction."""
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer index.")
        action = int(action)
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count}).")
        self.replay.add(
            state,
            action,
            reward,
            next_state,
            bootstrap_terminal,
            next_action_mask,
        )
        self.env_steps += 1
        self.epsilon = self.epsilon_for_environment_steps(self.env_steps)

    def can_learn(self) -> bool:
        return len(self.replay) >= self.batch_size

    def learn(self) -> dict[str, float | bool] | None:
        """Run one Double DQN update, or return ``None`` before replay warm-up."""
        if not self.can_learn():
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        bootstrap_terminals = torch.as_tensor(
            batch["bootstrap_terminals"], dtype=torch.bool, device=self.device
        )
        next_action_masks = torch.as_tensor(
            batch["next_action_masks"], dtype=torch.bool, device=self.device
        )

        predicted_q = self._selected_action_q(
            self.online_network, states, actions
        )
        with torch.no_grad():
            next_online_q = self._all_action_q(
                self.online_network, next_states
            ).masked_fill(
                ~next_action_masks, INVALID_ACTION_VALUE
            )
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_q = self._selected_action_q(
                self.target_network,
                next_states,
                next_actions.squeeze(1),
            )
            bootstrap_mask = (~bootstrap_terminals).to(dtype=torch.float32)
            target_q = rewards + self.gamma * bootstrap_mask * next_q

        loss = self.loss_function(predicted_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(), self.max_grad_norm
            )
            gradient_norm_value = float(gradient_norm.detach().item())
        else:
            gradient_norm_value = float("nan")
        self.optimizer.step()

        self.learning_steps += 1
        target_synchronized = (
            self.learning_steps % self.target_update_interval == 0
        )
        if target_synchronized:
            self.synchronize_target_network()
        return {
            "loss": float(loss.detach().item()),
            "mean_predicted_q": float(predicted_q.detach().mean().item()),
            "mean_target_q": float(target_q.detach().mean().item()),
            "gradient_norm": gradient_norm_value,
            "epsilon": float(self.epsilon),
            "target_synchronized": target_synchronized,
        }

    def synchronize_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

    def _configuration_state(self) -> dict[str, Any]:
        return {
            "state_dimension": self.state_dimension,
            "action_count": self.action_count,
            "hidden_dimensions": list(self.hidden_dimensions),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_schedule": self.epsilon_schedule,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "learning_starts": self.learning_starts,
            "target_update_interval": self.target_update_interval,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "action_mask_mode": ACTION_MASK_MODE,
            "q_network_mode": self.q_network_mode,
            "action_score_chunk_size": self.action_score_chunk_size,
            "action_features": (
                None
                if self.action_features is None
                else self.action_features.tolist()
            ),
        }

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        cuda_states: list[torch.Tensor] = []
        if torch.cuda.is_available():
            cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_cuda": cuda_states,
        }

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu"))
        cuda_states = state.get("torch_cuda", [])
        if torch.cuda.is_available() and cuda_states:
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                raise ValueError(
                    "Checkpoint CUDA RNG-state count does not match available CUDA devices."
                )
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, device="cpu") for item in cuda_states]
            )

    def agent_state(self) -> dict[str, Any]:
        """Return the complete algorithm state for a resumable checkpoint."""
        return {
            "schema_version": AGENT_STATE_SCHEMA_VERSION,
            "configuration": self._configuration_state(),
            "online_network": copy.deepcopy(self.online_network.state_dict()),
            "target_network": copy.deepcopy(self.target_network.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "epsilon": float(self.epsilon),
            "env_steps": int(self.env_steps),
            "learning_steps": int(self.learning_steps),
            "replay": self.replay.state_dict(),
            "rng_state": self._rng_state(),
        }

    def _validate_checkpoint_architecture(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported Double DQN agent-state schema.")
        configuration = state.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("Agent checkpoint is missing its configuration.")
        expected = {
            "state_dimension": self.state_dimension,
            "action_count": self.action_count,
            "hidden_dimensions": list(self.hidden_dimensions),
            "action_mask_mode": ACTION_MASK_MODE,
            "q_network_mode": self.q_network_mode,
            "action_score_chunk_size": self.action_score_chunk_size,
            "action_features": (
                None
                if self.action_features is None
                else self.action_features.tolist()
            ),
        }
        for name, value in expected.items():
            if configuration.get(name) != value:
                raise ValueError(
                    f"Checkpoint {name} does not match the configured agent."
                )

    def load_training_state(
        self,
        state: Mapping[str, Any],
        *,
        restore_rng: bool = True,
    ) -> None:
        """Restore networks, optimizer, counters, replay, epsilon, and RNGs."""
        self._validate_checkpoint_architecture(state)
        configuration = state["configuration"]
        current_configuration = self._configuration_state()
        for name, current_value in current_configuration.items():
            if configuration.get(name) != current_value:
                raise ValueError(
                    f"Checkpoint {name} does not match the training configuration."
                )

        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for name, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[name] = value.to(self.device)

        env_steps = int(state["env_steps"])
        learning_steps = int(state["learning_steps"])
        if env_steps < 0 or learning_steps < 0:
            raise ValueError("Checkpoint step counters must be nonnegative.")
        epsilon = _finite_float("checkpoint epsilon", state["epsilon"])
        expected_epsilon = self.epsilon_for_environment_steps(env_steps)
        if not np.isclose(epsilon, expected_epsilon, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "Checkpoint epsilon is inconsistent with its environment-step schedule."
            )
        self.epsilon = expected_epsilon
        self.env_steps = env_steps
        self.learning_steps = learning_steps
        self.replay.load_state_dict(state["replay"])
        if self.replay.size and np.any(
            self.replay.actions[: self.replay.size] >= self.action_count
        ):
            raise ValueError("Checkpoint replay contains an action outside the joint action space.")
        if restore_rng:
            self._restore_rng_state(state["rng_state"])

        self.online_network.train()
        self.target_network.eval()

    def load_inference_state(self, state: Mapping[str, Any]) -> None:
        """Load only policy weights for deterministic greedy inference.

        Optimizer, replay, counters, epsilon, and RNG state are intentionally not
        restored.  The target copy is synchronized only to keep the object
        internally coherent; inference actions always use the online network.
        """
        self._validate_checkpoint_architecture(state)
        self.online_network.load_state_dict(state["online_network"])
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.epsilon = 0.0
        self.online_network.eval()
        self.target_network.eval()


ReplayBuffer = UniformReplayBuffer


__all__ = [
    "ACTION_MASK_MODE",
    "DoubleDQNAgent",
    "QNetwork",
    "ReplayBuffer",
    "UniformReplayBuffer",
    "set_global_seed",
]

