"""Pure PyTorch discrete PPO components for the cascade controller."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from itertools import chain
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


PPO_AGENT_STATE_SCHEMA_VERSION = 3
ACTION_MASK_MODE = "environment_transition_reachability_and_monthly_limits_v2"
INVALID_ACTION_LOGIT = -1.0e9
ACTION_CONDITIONED_ACTOR_MODE = "action_conditioned"


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def set_ppo_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch CPU/CUDA RNGs deterministically."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be nonnegative.")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DiscretePPOActor(nn.Module):
    """Action-conditioned categorical actor over a fixed joint-action grid.

    The policy still observes the original five-component state.  The two
    action features describe each candidate target-level pair, so the actor
    learns one shared score function instead of 3,600 unrelated output heads.
    """

    def __init__(
        self,
        state_dimension: int,
        hidden_dimensions: Sequence[int],
        action_features: Sequence[Sequence[float]] | np.ndarray,
        action_score_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", dimension)
            for index, dimension in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("Actor hidden_dimensions cannot be empty.")
        features = np.asarray(action_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 2:
            raise ValueError("action_features must have shape [action_count, 2].")
        if len(features) == 0 or np.any(~np.isfinite(features)):
            raise ValueError("action_features must be nonempty and finite.")
        if np.any(features < 0.0) or np.any(features > 1.0):
            raise ValueError("action_features must lie in [0, 1].")
        self.action_count = int(features.shape[0])
        self.action_feature_dimension = int(features.shape[1])
        self.action_score_chunk_size = _positive_int(
            "action_score_chunk_size", action_score_chunk_size
        )
        self.register_buffer(
            "action_features",
            torch.as_tensor(features.copy(), dtype=torch.float32),
        )
        layers: list[nn.Module] = []
        hidden_linears: list[nn.Linear] = []
        previous = self.state_dimension
        for dimension in self.hidden_dimensions:
            linear = nn.Linear(previous, dimension)
            hidden_linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            previous = dimension
        self.trunk = nn.Sequential(*layers)
        self.embedding_dimension = min(128, previous)
        self.state_action_embedding = nn.Linear(
            previous, self.embedding_dimension
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_feature_dimension, self.embedding_dimension),
            nn.Tanh(),
            nn.Linear(self.embedding_dimension, self.embedding_dimension),
            nn.Tanh(),
        )
        self.action_bias = nn.Sequential(
            nn.Linear(self.action_feature_dimension, self.embedding_dimension),
            nn.Tanh(),
            nn.Linear(self.embedding_dimension, 1),
        )
        for linear in hidden_linears:
            _orthogonal_linear(linear, math.sqrt(2.0))
        _orthogonal_linear(self.state_action_embedding, 0.01)
        _orthogonal_linear(self.action_encoder[0], math.sqrt(2.0))
        _orthogonal_linear(self.action_encoder[2], 0.01)
        _orthogonal_linear(self.action_bias[0], math.sqrt(2.0))
        _orthogonal_linear(self.action_bias[2], 0.01)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded_states = self.trunk(states)
        state_embedding = self.state_action_embedding(encoded_states)
        pieces: list[torch.Tensor] = []
        scale = float(self.embedding_dimension) ** 0.5
        for start in range(0, self.action_count, self.action_score_chunk_size):
            features = self.action_features[
                start : start + self.action_score_chunk_size
            ]
            action_embedding = self.action_encoder(features)
            interaction = state_embedding @ action_embedding.transpose(0, 1)
            action_bias = self.action_bias(features).transpose(0, 1)
            pieces.append(interaction / scale + action_bias)
        return torch.cat(pieces, dim=1)


class PPOCritic(nn.Module):
    """Separate state-value network."""

    def __init__(self, state_dimension: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", dimension)
            for index, dimension in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("Critic hidden_dimensions cannot be empty.")
        layers: list[nn.Module] = []
        hidden_linears: list[nn.Linear] = []
        previous = self.state_dimension
        for dimension in self.hidden_dimensions:
            linear = nn.Linear(previous, dimension)
            hidden_linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            previous = dimension
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)
        for linear in hidden_linears:
            _orthogonal_linear(linear, math.sqrt(2.0))
        _orthogonal_linear(self.output, 1.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(states)).squeeze(-1)


@dataclass(frozen=True)
class PPORolloutBatch:
    """One complete synchronous rollout with explicit boundary semantics."""

    states: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_allowed: np.ndarray
    action_masks: np.ndarray

    def validate(
        self, *, state_dimension: int, action_count: int | None = None
    ) -> tuple[int, int]:
        states = np.asarray(self.states)
        if states.ndim != 3 or states.shape[2] != state_dimension:
            raise ValueError(
                "rollout states must have shape [n_steps, num_envs, state_dimension]."
            )
        n_steps, num_envs, _ = states.shape
        expected = (n_steps, num_envs)
        arrays = {
            "actions": self.actions,
            "old_log_probs": self.old_log_probs,
            "rewards": self.rewards,
            "values": self.values,
            "next_values": self.next_values,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "bootstrap_allowed": self.bootstrap_allowed,
        }
        for name, value in arrays.items():
            if np.asarray(value).shape != expected:
                raise ValueError(
                    f"rollout {name} must have shape {expected}; "
                    f"received {np.asarray(value).shape}."
                )
        numeric = (
            states,
            np.asarray(self.old_log_probs),
            np.asarray(self.rewards),
            np.asarray(self.values),
            np.asarray(self.next_values),
        )
        if any(np.any(~np.isfinite(value)) for value in numeric):
            raise ValueError("rollout numeric arrays must be finite.")
        if not np.issubdtype(np.asarray(self.actions).dtype, np.integer):
            raise ValueError("rollout actions must be integers.")
        for name in ("terminated", "truncated", "bootstrap_allowed"):
            if np.asarray(arrays[name]).dtype != np.bool_:
                raise ValueError(f"rollout {name} must be boolean.")
        action_masks = np.asarray(self.action_masks)
        if (
            action_masks.ndim != 3
            or action_masks.shape[:2] != expected
            or action_masks.dtype != np.bool_
        ):
            raise ValueError(
                "rollout action_masks must be boolean with shape "
                "[n_steps, num_envs, action_count]."
            )
        if action_count is not None and action_masks.shape[2] != action_count:
            raise ValueError(
                f"rollout action_masks must contain {action_count} actions."
            )
        if np.any(~action_masks.any(axis=2)):
            raise ValueError("Every rollout state must have a feasible action.")
        actions = np.asarray(self.actions, dtype=np.int64)
        if np.any(actions < 0) or np.any(actions >= action_masks.shape[2]):
            raise ValueError("rollout contains an action outside its mask.")
        if np.any(~np.take_along_axis(action_masks, actions[..., None], axis=2)[..., 0]):
            raise ValueError("rollout contains an action marked physically infeasible.")
        terminated = np.asarray(self.terminated, dtype=np.bool_)
        bootstrap_allowed = np.asarray(self.bootstrap_allowed, dtype=np.bool_)
        if np.any(terminated & bootstrap_allowed):
            raise ValueError("A terminated transition cannot allow bootstrap.")
        if np.any((~terminated) & (~bootstrap_allowed)):
            raise ValueError("Every nonterminated transition must allow bootstrap.")
        return n_steps, num_envs


def compute_gae(
    rollout: PPORolloutBatch,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with separate bootstrap and trace-continuation masks."""
    gamma_value = _finite_float("gamma", gamma)
    lambda_value = _finite_float("gae_lambda", gae_lambda)
    if not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1].")
    n_steps, num_envs = rollout.validate(
        state_dimension=np.asarray(rollout.states).shape[-1]
    )
    rewards = np.asarray(rollout.rewards, dtype=np.float64)
    values = np.asarray(rollout.values, dtype=np.float64)
    next_values = np.asarray(rollout.next_values, dtype=np.float64)
    terminated = np.asarray(rollout.terminated, dtype=np.bool_)
    truncated = np.asarray(rollout.truncated, dtype=np.bool_)
    bootstrap_allowed = np.asarray(rollout.bootstrap_allowed, dtype=np.bool_)
    bootstrap_mask = bootstrap_allowed.astype(np.float64)
    continuation_mask = (~(terminated | truncated)).astype(np.float64)

    advantages = np.zeros((n_steps, num_envs), dtype=np.float64)
    next_advantage = np.zeros(num_envs, dtype=np.float64)
    for step in range(n_steps - 1, -1, -1):
        delta = (
            rewards[step]
            + gamma_value * bootstrap_mask[step] * next_values[step]
            - values[step]
        )
        next_advantage = (
            delta
            + gamma_value
            * lambda_value
            * continuation_mask[step]
            * next_advantage
        )
        advantages[step] = next_advantage
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        norm = parameter.grad.detach().norm(2).to(dtype=torch.float64, device="cpu")
        squared += norm * norm
    return float(torch.sqrt(squared).item()) if found else 0.0


class DiscretePPOAgent:
    """Separate Actor/Critic PPO agent with one shared Adam optimizer."""

    def __init__(
        self,
        *,
        state_dimension: int,
        action_count: int,
        action_features: Sequence[Sequence[float]] | np.ndarray,
        action_score_chunk_size: int,
        actor_hidden_dimensions: Sequence[int],
        critic_hidden_dimensions: Sequence[int],
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        normalized_entropy_coefficient: float,
        value_loss_coefficient: float,
        max_gradient_norm: float,
        minibatch_size: int,
        update_epochs: int,
        advantage_normalization: bool,
        adam_epsilon: float,
        seed: int,
        device: str | torch.device,
    ) -> None:
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.action_count = _positive_int("action_count", action_count)
        features = np.asarray(action_features, dtype=np.float32)
        if features.shape != (self.action_count, 2):
            raise ValueError(
                f"action_features must have shape ({self.action_count}, 2)."
            )
        if np.any(~np.isfinite(features)):
            raise ValueError("action_features must be finite.")
        self.action_features = features.copy()
        self.action_feature_dimension = 2
        action_feature_digest = hashlib.sha256()
        action_feature_digest.update(str(features.shape).encode("ascii"))
        action_feature_digest.update(features.tobytes(order="C"))
        self.action_feature_sha256 = action_feature_digest.hexdigest()
        self.action_score_chunk_size = _positive_int(
            "action_score_chunk_size", action_score_chunk_size
        )
        self.actor_mode = ACTION_CONDITIONED_ACTOR_MODE
        self.actor_hidden_dimensions = tuple(actor_hidden_dimensions)
        self.critic_hidden_dimensions = tuple(critic_hidden_dimensions)
        self.learning_rate = _finite_float("learning_rate", learning_rate)
        self.gamma = _finite_float("gamma", gamma)
        self.gae_lambda = _finite_float("gae_lambda", gae_lambda)
        self.clip_range = _finite_float("clip_range", clip_range)
        self.normalized_entropy_coefficient = _finite_float(
            "normalized_entropy_coefficient", normalized_entropy_coefficient
        )
        self.value_loss_coefficient = _finite_float(
            "value_loss_coefficient", value_loss_coefficient
        )
        self.max_gradient_norm = _finite_float(
            "max_gradient_norm", max_gradient_norm
        )
        self.minibatch_size = _positive_int("minibatch_size", minibatch_size)
        self.update_epochs = _positive_int("update_epochs", update_epochs)
        if not isinstance(advantage_normalization, bool):
            raise TypeError("advantage_normalization must be boolean.")
        self.advantage_normalization = advantage_normalization
        self.adam_epsilon = _finite_float("adam_epsilon", adam_epsilon)
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1].")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must lie in (0, 1).")
        if self.normalized_entropy_coefficient < 0.0:
            raise ValueError("normalized_entropy_coefficient must be nonnegative.")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("value_loss_coefficient must be nonnegative.")
        if self.max_gradient_norm <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("gradient norm and Adam epsilon must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        self.device = torch.device(device)
        self.maximum_entropy = float(math.log(self.action_count))

        set_ppo_global_seed(self.seed)
        self.actor = DiscretePPOActor(
            self.state_dimension,
            self.actor_hidden_dimensions,
            self.action_features,
            self.action_score_chunk_size,
        ).to(self.device)
        self.critic = PPOCritic(
            self.state_dimension,
            self.critic_hidden_dimensions,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.learning_rate,
            eps=self.adam_epsilon,
        )

    def _state_tensor(self, states: np.ndarray | Sequence[float]) -> torch.Tensor:
        array = np.asarray(states, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2 or array.shape[1] != self.state_dimension:
            raise ValueError(
                f"states must have shape [batch, {self.state_dimension}]."
            )
        if np.any(~np.isfinite(array)):
            raise ValueError("states must be finite.")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def act(
        self,
        states: np.ndarray | Sequence[float],
        *,
        deterministic: bool,
        action_masks: np.ndarray | Sequence[Sequence[bool]] | Sequence[bool],
    ) -> dict[str, np.ndarray]:
        """Sample or greedily select from physics-feasible categorical actions."""
        tensor = self._state_tensor(states)
        masks = np.asarray(action_masks)
        if masks.ndim == 1:
            masks = masks[np.newaxis, :]
        expected = (tensor.shape[0], self.action_count)
        if masks.shape != expected or masks.dtype != np.bool_:
            raise ValueError(
                f"action_masks must be boolean with shape {expected}."
            )
        if np.any(~masks.any(axis=1)):
            raise ValueError("Every PPO state must have a feasible action.")
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.actor(tensor)
            masked_logits = logits.masked_fill(~mask_tensor, INVALID_ACTION_LOGIT)
            distribution = Categorical(logits=masked_logits)
            actions = (
                masked_logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probs = distribution.log_prob(actions)
            values = self.critic(tensor)
            entropies = distribution.entropy()
            max_probabilities = distribution.probs.max(dim=-1).values
        return {
            "actions": actions.cpu().numpy().astype(np.int64, copy=False),
            "log_probs": log_probs.cpu().numpy().astype(np.float32, copy=False),
            "values": values.cpu().numpy().astype(np.float32, copy=False),
            "raw_entropies": entropies.cpu().numpy().astype(np.float32, copy=False),
            "max_action_probabilities": max_probabilities.cpu()
            .numpy()
            .astype(np.float32, copy=False),
        }

    def predict_values(self, states: np.ndarray | Sequence[float]) -> np.ndarray:
        tensor = self._state_tensor(states)
        with torch.no_grad():
            values = self.critic(tensor)
        return values.cpu().numpy().astype(np.float32, copy=False)

    def update(self, rollout: PPORolloutBatch) -> dict[str, float | int]:
        """Run all configured PPO epochs over one complete on-policy rollout."""
        n_steps, num_envs = rollout.validate(
            state_dimension=self.state_dimension,
            action_count=self.action_count,
        )
        rollout_size = n_steps * num_envs
        if rollout_size % self.minibatch_size != 0:
            raise ValueError(
                "rollout size must be exactly divisible by minibatch_size."
            )
        actions_array = np.asarray(rollout.actions, dtype=np.int64)
        if np.any(actions_array < 0) or np.any(actions_array >= self.action_count):
            raise ValueError("rollout contains an action outside the action space.")
        advantages, returns = compute_gae(
            rollout,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        flat_states = np.asarray(rollout.states, dtype=np.float32).reshape(
            rollout_size, self.state_dimension
        )
        flat_actions = actions_array.reshape(rollout_size)
        flat_old_log_probs = np.asarray(
            rollout.old_log_probs, dtype=np.float32
        ).reshape(rollout_size)
        flat_old_values = np.asarray(rollout.values, dtype=np.float32).reshape(
            rollout_size
        )
        flat_action_masks = np.asarray(
            rollout.action_masks, dtype=np.bool_
        ).reshape(rollout_size, self.action_count)
        flat_advantages = advantages.reshape(rollout_size)
        flat_returns = returns.reshape(rollout_size)
        advantage_mean_before = float(flat_advantages.mean())
        advantage_std_before = float(flat_advantages.std(ddof=0))
        if self.advantage_normalization:
            flat_advantages = (
                flat_advantages - advantage_mean_before
            ) / max(advantage_std_before, 1.0e-8)
        advantage_mean_after = float(flat_advantages.mean())
        advantage_std_after = float(flat_advantages.std(ddof=0))

        states = torch.as_tensor(flat_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(flat_actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(
            flat_old_log_probs, dtype=torch.float32, device=self.device
        )
        action_masks = torch.as_tensor(
            flat_action_masks, dtype=torch.bool, device=self.device
        )
        normalized_advantages = torch.as_tensor(
            flat_advantages, dtype=torch.float32, device=self.device
        )
        return_targets = torch.as_tensor(
            flat_returns, dtype=torch.float32, device=self.device
        )
        metric_totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "raw_entropy": 0.0,
            "normalized_entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "effective_action_count": 0.0,
            "max_action_probability": 0.0,
            "gradient_norm": 0.0,
            "actor_gradient_norm": 0.0,
            "critic_gradient_norm": 0.0,
        }
        ratio_max = 0.0
        minibatch_updates = 0
        actor_parameters = list(self.actor.parameters())
        critic_parameters = list(self.critic.parameters())
        all_parameters = [*actor_parameters, *critic_parameters]

        self.actor.train()
        self.critic.train()
        for _ in range(self.update_epochs):
            indices = torch.randperm(rollout_size, device="cpu")
            for start in range(0, rollout_size, self.minibatch_size):
                batch_indices = indices[start : start + self.minibatch_size].to(
                    self.device
                )
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = normalized_advantages[batch_indices]
                batch_returns = return_targets[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                logits = self.actor(batch_states)
                masked_logits = logits.masked_fill(
                    ~batch_action_masks, INVALID_ACTION_LOGIT
                )
                distribution = Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(batch_actions)
                new_values = self.critic(batch_states)
                raw_entropy = distribution.entropy()
                normalized_entropy = raw_entropy / self.maximum_entropy
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_range,
                    1.0 + self.clip_range,
                ) * batch_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.mean((new_values - batch_returns) ** 2)
                entropy_mean = raw_entropy.mean()
                normalized_entropy_mean = normalized_entropy.mean()
                total_loss = (
                    policy_loss
                    + self.value_loss_coefficient * value_loss
                    - self.normalized_entropy_coefficient
                    * normalized_entropy_mean
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("PPO produced a non-finite total loss.")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                actor_gradient_norm = _gradient_norm(actor_parameters)
                critic_gradient_norm = _gradient_norm(critic_parameters)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        all_parameters,
                        self.max_gradient_norm,
                    ).detach().item()
                )
                if not np.isfinite(gradient_norm):
                    raise FloatingPointError("PPO produced a non-finite gradient norm.")
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range)
                        .to(dtype=torch.float32)
                        .mean()
                    )
                    ratio_std = ratio.std(unbiased=False)
                    effective_action_count = torch.exp(raw_entropy).mean()
                    max_action_probability = distribution.probs.max(
                        dim=-1
                    ).values.mean()
                    ratio_max = max(ratio_max, float(ratio.max().item()))
                current = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "raw_entropy": entropy_mean,
                    "normalized_entropy": normalized_entropy_mean,
                    "total_loss": total_loss,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "ratio_mean": ratio.mean(),
                    "ratio_std": ratio_std,
                    "effective_action_count": effective_action_count,
                    "max_action_probability": max_action_probability,
                }
                for name, value in current.items():
                    metric_totals[name] += float(value.detach().item())
                metric_totals["gradient_norm"] += gradient_norm
                metric_totals["actor_gradient_norm"] += actor_gradient_norm
                metric_totals["critic_gradient_norm"] += critic_gradient_norm
                minibatch_updates += 1

        if minibatch_updates == 0:
            raise RuntimeError("PPO update performed no minibatch optimizer steps.")
        metrics: dict[str, float | int] = {
            name: total / minibatch_updates
            for name, total in metric_totals.items()
        }
        metrics.update(
            {
                "ratio_max": ratio_max,
                "advantage_mean_before_normalization": advantage_mean_before,
                "advantage_std_before_normalization": advantage_std_before,
                "advantage_mean_after_normalization": advantage_mean_after,
                "advantage_std_after_normalization": advantage_std_after,
                "return_mean": float(flat_returns.mean()),
                "return_std": float(flat_returns.std(ddof=0)),
                "old_value_prediction_mean": float(flat_old_values.mean()),
                "old_value_prediction_std": float(flat_old_values.std(ddof=0)),
                "unique_sampled_actions": int(np.unique(flat_actions).size),
                "rollout_size": rollout_size,
                "minibatch_updates": minibatch_updates,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "mean_feasible_action_count": float(
                    flat_action_masks.sum(axis=1).mean()
                ),
                "min_feasible_action_count": int(
                    flat_action_masks.sum(axis=1).min()
                ),
            }
        )
        with torch.no_grad():
            final_values = self.critic(states).cpu().numpy()
        residual_variance = float(np.var(flat_returns - final_values))
        return_variance = float(np.var(flat_returns))
        metrics["final_value_prediction_mean"] = float(final_values.mean())
        metrics["final_value_prediction_std"] = float(final_values.std(ddof=0))
        metrics["explained_variance"] = (
            float("nan")
            if return_variance <= 1.0e-12
            else float(1.0 - residual_variance / return_variance)
        )
        return metrics

    def configuration_state(self) -> dict[str, Any]:
        return {
            "state_dimension": self.state_dimension,
            "action_count": self.action_count,
            "actor_mode": self.actor_mode,
            "action_feature_dimension": self.action_feature_dimension,
            "action_feature_sha256": self.action_feature_sha256,
            "action_score_chunk_size": self.action_score_chunk_size,
            "actor_hidden_dimensions": list(self.actor_hidden_dimensions),
            "critic_hidden_dimensions": list(self.critic_hidden_dimensions),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "normalized_entropy_coefficient": self.normalized_entropy_coefficient,
            "value_loss_coefficient": self.value_loss_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "advantage_normalization": self.advantage_normalization,
            "adam_epsilon": self.adam_epsilon,
            "seed": self.seed,
            "action_mask_mode": ACTION_MASK_MODE,
        }

    @staticmethod
    def rng_state() -> dict[str, Any]:
        cuda_states: list[torch.Tensor] = []
        if torch.cuda.is_available():
            cuda_states = [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_cuda": cuda_states,
        }

    @staticmethod
    def restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"], device="cpu"))
        cuda_states = list(state.get("torch_cuda", []))
        if torch.cuda.is_available() and cuda_states:
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    "Checkpoint CUDA RNG count does not match visible CUDA devices."
                )
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, device="cpu") for item in cuda_states]
            )

    def training_state(self) -> dict[str, Any]:
        return {
            "schema_version": PPO_AGENT_STATE_SCHEMA_VERSION,
            "configuration": self.configuration_state(),
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "rng_state": self.rng_state(),
        }

    def _validate_state_configuration(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != PPO_AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported PPO agent-state schema.")
        configuration = state.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("PPO state is missing configuration metadata.")
        expected = self.configuration_state()
        for name, value in expected.items():
            if configuration.get(name) != value:
                raise ValueError(f"Checkpoint PPO {name} does not match configuration.")

    def load_training_state(
        self,
        state: Mapping[str, Any],
        *,
        restore_rng: bool,
    ) -> None:
        self._validate_state_configuration(state)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for name, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[name] = value.to(self.device)
        if restore_rng:
            self.restore_rng_state(state["rng_state"])
        self.actor.train()
        self.critic.train()

    def load_inference_state(self, state: Mapping[str, Any]) -> None:
        self._validate_state_configuration(state)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor.eval()
        self.critic.eval()


__all__ = [
    "ACTION_CONDITIONED_ACTOR_MODE",
    "ACTION_MASK_MODE",
    "DiscretePPOActor",
    "DiscretePPOAgent",
    "PPOCritic",
    "PPORolloutBatch",
    "compute_gae",
    "set_ppo_global_seed",
]

