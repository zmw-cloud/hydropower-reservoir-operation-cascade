"""Autoregressive squashed-Gaussian PPO for continuous cascade targets."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from itertools import chain
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .continuous_ppo_adapter import (
    ACTION_GEOMETRY_DIMENSION,
    NUOZHADU_GEOMETRY_DIMENSION,
    XIAOWAN_GEOMETRY_DIMENSION,
)


CONTINUOUS_PPO_AGENT_STATE_SCHEMA_VERSION = 2
CONTINUOUS_ACTION_MODE = "geometry_conditioned_autoregressive_squashed_gaussian"
CONTINUOUS_ACTION_DIMENSION = 2


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


def set_continuous_ppo_global_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be nonnegative.")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class AutoregressiveSquashedGaussianActor(nn.Module):
    """Five-state actor for two bounded, autoregressive action fractions."""

    def __init__(
        self,
        state_dimension: int,
        hidden_dimensions: Sequence[int],
        *,
        initial_log_std: float,
        min_log_std: float,
        max_log_std: float,
        squash_epsilon: float,
    ) -> None:
        super().__init__()
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", value)
            for index, value in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("Actor hidden_dimensions cannot be empty.")
        self.initial_log_std = _finite_float("initial_log_std", initial_log_std)
        self.min_log_std = _finite_float("min_log_std", min_log_std)
        self.max_log_std = _finite_float("max_log_std", max_log_std)
        self.squash_epsilon = _finite_float("squash_epsilon", squash_epsilon)
        if not self.min_log_std < self.max_log_std:
            raise ValueError("min_log_std must be less than max_log_std.")
        if not self.min_log_std <= self.initial_log_std <= self.max_log_std:
            raise ValueError("initial_log_std must lie inside the configured bounds.")
        if not 0.0 < self.squash_epsilon < 1.0e-2:
            raise ValueError("squash_epsilon must lie in (0, 1e-2).")

        layers: list[nn.Module] = []
        hidden_linears: list[nn.Linear] = []
        previous = self.state_dimension
        for dimension in self.hidden_dimensions:
            linear = nn.Linear(previous, dimension)
            hidden_linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            previous = dimension
        self.state_trunk = nn.Sequential(*layers)
        conditional_hidden = min(128, previous)
        self.xiaowan_conditioner = nn.Sequential(
            nn.Linear(
                previous + XIAOWAN_GEOMETRY_DIMENSION,
                conditional_hidden,
            ),
            nn.Tanh(),
            nn.Linear(conditional_hidden, 1),
        )
        self.nuozhadu_conditioner = nn.Sequential(
            nn.Linear(
                previous + 1 + NUOZHADU_GEOMETRY_DIMENSION,
                conditional_hidden,
            ),
            nn.Tanh(),
            nn.Linear(conditional_hidden, 1),
        )
        self.log_std = nn.Parameter(
            torch.full((CONTINUOUS_ACTION_DIMENSION,), self.initial_log_std)
        )

        for linear in hidden_linears:
            _orthogonal_linear(linear, math.sqrt(2.0))
        _orthogonal_linear(self.xiaowan_conditioner[0], math.sqrt(2.0))
        _orthogonal_linear(self.xiaowan_conditioner[2], 0.01)
        _orthogonal_linear(self.nuozhadu_conditioner[0], math.sqrt(2.0))
        _orthogonal_linear(self.nuozhadu_conditioner[2], 0.01)

    def _std(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_std, self.min_log_std, self.max_log_std))

    @staticmethod
    def fractions_from_latent(latent_actions: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.tanh(latent_actions) + 1.0)

    def _squashed_log_prob(
        self,
        distribution: Normal,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        tanh_value = torch.tanh(latent)
        log_jacobian = torch.log(
            torch.clamp(1.0 - tanh_value.square(), min=self.squash_epsilon)
        ) - math.log(2.0)
        return distribution.log_prob(latent) - log_jacobian

    def evaluate_latent_actions(
        self,
        states: torch.Tensor,
        latent_actions: torch.Tensor,
        action_geometry: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if latent_actions.ndim != 2 or latent_actions.shape[1] != 2:
            raise ValueError("latent_actions must have shape [batch, 2].")
        if action_geometry.shape != (len(states), ACTION_GEOMETRY_DIMENSION):
            raise ValueError(
                "action_geometry must have shape [batch, 7]."
            )
        encoded = self.state_trunk(states)
        std = self._std()

        xiaowan_input = torch.cat(
            (
                encoded,
                action_geometry[:, :XIAOWAN_GEOMETRY_DIMENSION],
            ),
            dim=1,
        )
        xiaowan_mean = self.xiaowan_conditioner(
            xiaowan_input
        ).squeeze(-1)
        xiaowan_distribution = Normal(xiaowan_mean, std[0])
        xiaowan_latent = latent_actions[:, 0]
        xiaowan_fraction = self.fractions_from_latent(xiaowan_latent)

        downstream_input = torch.cat(
            (
                encoded,
                xiaowan_fraction.unsqueeze(-1),
                action_geometry[:, XIAOWAN_GEOMETRY_DIMENSION:],
            ),
            dim=1,
        )
        nuozhadu_mean = self.nuozhadu_conditioner(downstream_input).squeeze(-1)
        nuozhadu_distribution = Normal(nuozhadu_mean, std[1])
        nuozhadu_latent = latent_actions[:, 1]

        component_log_probs = torch.stack(
            (
                self._squashed_log_prob(xiaowan_distribution, xiaowan_latent),
                self._squashed_log_prob(nuozhadu_distribution, nuozhadu_latent),
            ),
            dim=1,
        )
        log_prob = component_log_probs.sum(dim=1)
        fractions = self.fractions_from_latent(latent_actions)
        means = torch.stack((xiaowan_mean, nuozhadu_mean), dim=1)
        return {
            "latent_actions": latent_actions,
            "component_log_probs": component_log_probs,
            "log_probs": log_prob,
            "sample_entropies": -log_prob,
            "action_fractions": fractions,
            "latent_means": means,
            "action_std": std.expand(len(states), -1),
        }

    def act_xiaowan(
        self,
        states: torch.Tensor,
        xiaowan_geometry: torch.Tensor,
        *,
        deterministic: bool,
        xiaowan_active: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if xiaowan_geometry.shape != (
            len(states), XIAOWAN_GEOMETRY_DIMENSION
        ):
            raise ValueError(
                "xiaowan_geometry must have shape [batch, 3]."
            )
        encoded = self.state_trunk(states)
        std = self._std()
        xiaowan_mean = self.xiaowan_conditioner(
            torch.cat((encoded, xiaowan_geometry), dim=1)
        ).squeeze(-1)
        xiaowan_distribution = Normal(xiaowan_mean, std[0])
        xiaowan_latent = xiaowan_mean if deterministic else xiaowan_distribution.sample()
        if xiaowan_active is not None:
            if xiaowan_active.shape != xiaowan_latent.shape:
                raise ValueError("xiaowan_active must match the batch dimension.")
            xiaowan_latent = torch.where(
                xiaowan_active,
                xiaowan_latent,
                torch.zeros_like(xiaowan_latent),
            )
        xiaowan_fraction = self.fractions_from_latent(xiaowan_latent)
        return {
            "latent": xiaowan_latent,
            "fraction": xiaowan_fraction,
            "mean": xiaowan_mean,
            "std": std[0].expand(len(states)),
            "log_prob": self._squashed_log_prob(
                xiaowan_distribution, xiaowan_latent
            ),
        }

    def complete_action(
        self,
        states: torch.Tensor,
        xiaowan_latent: torch.Tensor,
        xiaowan_geometry: torch.Tensor,
        nuozhadu_geometry: torch.Tensor,
        *,
        deterministic: bool,
        nuozhadu_active: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if xiaowan_latent.shape != (len(states),):
            raise ValueError("xiaowan_latent must match the batch dimension.")
        if nuozhadu_geometry.shape != (
            len(states), NUOZHADU_GEOMETRY_DIMENSION
        ):
            raise ValueError(
                "nuozhadu_geometry must have shape [batch, 4]."
            )
        encoded = self.state_trunk(states)
        std = self._std()
        xiaowan_fraction = self.fractions_from_latent(xiaowan_latent)
        downstream_input = torch.cat(
            (
                encoded,
                xiaowan_fraction.unsqueeze(-1),
                nuozhadu_geometry,
            ),
            dim=1,
        )
        nuozhadu_mean = self.nuozhadu_conditioner(
            downstream_input
        ).squeeze(-1)
        distribution = Normal(nuozhadu_mean, std[1])
        nuozhadu_latent = (
            nuozhadu_mean if deterministic else distribution.sample()
        )
        if nuozhadu_active is not None:
            if nuozhadu_active.shape != nuozhadu_latent.shape:
                raise ValueError(
                    "nuozhadu_active must match the batch dimension."
                )
            nuozhadu_latent = torch.where(
                nuozhadu_active,
                nuozhadu_latent,
                torch.zeros_like(nuozhadu_latent),
            )
        latent_actions = torch.stack(
            (xiaowan_latent, nuozhadu_latent), dim=1
        )
        geometry = torch.cat(
            (xiaowan_geometry, nuozhadu_geometry), dim=1
        )
        return self.evaluate_latent_actions(
            states, latent_actions, geometry
        )


class ContinuousPPOCritic(nn.Module):
    def __init__(self, state_dimension: int, hidden_dimensions: Sequence[int]) -> None:
        super().__init__()
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.hidden_dimensions = tuple(
            _positive_int(f"hidden_dimensions[{index}]", value)
            for index, value in enumerate(hidden_dimensions)
        )
        if not self.hidden_dimensions:
            raise ValueError("Critic hidden_dimensions cannot be empty.")
        layers: list[nn.Module] = []
        linears: list[nn.Linear] = []
        previous = self.state_dimension
        for dimension in self.hidden_dimensions:
            linear = nn.Linear(previous, dimension)
            linears.append(linear)
            layers.extend((linear, nn.Tanh()))
            previous = dimension
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)
        for linear in linears:
            _orthogonal_linear(linear, math.sqrt(2.0))
        _orthogonal_linear(self.output, 1.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(states)).squeeze(-1)


@dataclass(frozen=True)
class ContinuousPPORolloutBatch:
    states: np.ndarray
    latent_actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_allowed: np.ndarray
    active_action_dimensions: np.ndarray
    action_geometry: np.ndarray

    def validate(self, *, state_dimension: int) -> tuple[int, int]:
        states = np.asarray(self.states)
        if states.ndim != 3 or states.shape[2] != state_dimension:
            raise ValueError(
                "rollout states must have shape [n_steps, num_envs, state_dimension]."
            )
        n_steps, num_envs, _ = states.shape
        expected = (n_steps, num_envs)
        latent = np.asarray(self.latent_actions)
        if latent.shape != (n_steps, num_envs, CONTINUOUS_ACTION_DIMENSION):
            raise ValueError("rollout latent_actions must have shape [n_steps, num_envs, 2].")
        arrays = {
            "old_log_probs": self.old_log_probs,
            "rewards": self.rewards,
            "values": self.values,
            "next_values": self.next_values,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "bootstrap_allowed": self.bootstrap_allowed,
        }
        active = np.asarray(self.active_action_dimensions)
        if active.shape != (n_steps, num_envs, CONTINUOUS_ACTION_DIMENSION):
            raise ValueError(
                "rollout active_action_dimensions must have shape [n_steps, num_envs, 2]."
            )
        if active.dtype != np.bool_:
            raise ValueError("rollout active_action_dimensions must be boolean.")
        geometry = np.asarray(self.action_geometry)
        if geometry.shape != (
            n_steps,
            num_envs,
            ACTION_GEOMETRY_DIMENSION,
        ):
            raise ValueError(
                "rollout action_geometry must have shape "
                "[n_steps, num_envs, 7]."
            )
        for name, value in arrays.items():
            if np.asarray(value).shape != expected:
                raise ValueError(f"rollout {name} must have shape {expected}.")
        numeric = [states, latent, geometry]
        numeric.extend(
            np.asarray(arrays[name])
            for name in ("old_log_probs", "rewards", "values", "next_values")
        )
        if any(np.any(~np.isfinite(value)) for value in numeric):
            raise ValueError("rollout numeric arrays must be finite.")
        for name in ("terminated", "truncated", "bootstrap_allowed"):
            if np.asarray(arrays[name]).dtype != np.bool_:
                raise ValueError(f"rollout {name} must be boolean.")
        terminated = np.asarray(self.terminated, dtype=np.bool_)
        bootstrap = np.asarray(self.bootstrap_allowed, dtype=np.bool_)
        if np.any(terminated & bootstrap):
            raise ValueError("A terminated transition cannot allow bootstrap.")
        if np.any((~terminated) & (~bootstrap)):
            raise ValueError("Every nonterminated transition must allow bootstrap.")
        return n_steps, num_envs


def compute_continuous_gae(
    rollout: ContinuousPPORolloutBatch,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_value = _finite_float("gamma", gamma)
    lambda_value = _finite_float("gae_lambda", gae_lambda)
    if not 0.0 <= gamma_value <= 1.0 or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("gamma and gae_lambda must lie in [0, 1].")
    n_steps, num_envs = rollout.validate(
        state_dimension=np.asarray(rollout.states).shape[-1]
    )
    rewards = np.asarray(rollout.rewards, dtype=np.float64)
    values = np.asarray(rollout.values, dtype=np.float64)
    next_values = np.asarray(rollout.next_values, dtype=np.float64)
    bootstrap = np.asarray(rollout.bootstrap_allowed, dtype=np.float64)
    continuation = (~(
        np.asarray(rollout.terminated, dtype=np.bool_)
        | np.asarray(rollout.truncated, dtype=np.bool_)
    )).astype(np.float64)
    advantages = np.zeros((n_steps, num_envs), dtype=np.float64)
    next_advantage = np.zeros(num_envs, dtype=np.float64)
    for step in range(n_steps - 1, -1, -1):
        delta = rewards[step] + gamma_value * bootstrap[step] * next_values[step] - values[step]
        next_advantage = delta + gamma_value * lambda_value * continuation[step] * next_advantage
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


class ContinuousPPOAgent:
    def __init__(
        self,
        *,
        state_dimension: int,
        actor_hidden_dimensions: Sequence[int],
        critic_hidden_dimensions: Sequence[int],
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        entropy_coefficient: float,
        value_loss_coefficient: float,
        max_gradient_norm: float,
        minibatch_size: int,
        update_epochs: int,
        target_kl: float,
        advantage_normalization: bool,
        adam_epsilon: float,
        initial_log_std: float,
        min_log_std: float,
        max_log_std: float,
        squash_epsilon: float,
        seed: int,
        device: str | torch.device,
    ) -> None:
        self.state_dimension = _positive_int("state_dimension", state_dimension)
        self.actor_hidden_dimensions = tuple(actor_hidden_dimensions)
        self.critic_hidden_dimensions = tuple(critic_hidden_dimensions)
        self.learning_rate = _finite_float("learning_rate", learning_rate)
        self.gamma = _finite_float("gamma", gamma)
        self.gae_lambda = _finite_float("gae_lambda", gae_lambda)
        self.clip_range = _finite_float("clip_range", clip_range)
        self.entropy_coefficient = _finite_float("entropy_coefficient", entropy_coefficient)
        self.value_loss_coefficient = _finite_float("value_loss_coefficient", value_loss_coefficient)
        self.max_gradient_norm = _finite_float("max_gradient_norm", max_gradient_norm)
        self.minibatch_size = _positive_int("minibatch_size", minibatch_size)
        self.update_epochs = _positive_int("update_epochs", update_epochs)
        self.target_kl = _finite_float("target_kl", target_kl)
        self.advantage_normalization = bool(advantage_normalization)
        self.adam_epsilon = _finite_float("adam_epsilon", adam_epsilon)
        self.initial_log_std = _finite_float("initial_log_std", initial_log_std)
        self.min_log_std = _finite_float("min_log_std", min_log_std)
        self.max_log_std = _finite_float("max_log_std", max_log_std)
        self.squash_epsilon = _finite_float("squash_epsilon", squash_epsilon)
        if self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("learning rate and Adam epsilon must be positive.")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in [0, 1].")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError("clip_range must lie in (0, 1).")
        if self.entropy_coefficient < 0.0 or self.value_loss_coefficient < 0.0:
            raise ValueError("loss coefficients must be nonnegative.")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive.")
        if self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        self.device = torch.device(device)

        set_continuous_ppo_global_seed(self.seed)
        self.actor = AutoregressiveSquashedGaussianActor(
            self.state_dimension,
            self.actor_hidden_dimensions,
            initial_log_std=self.initial_log_std,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
            squash_epsilon=self.squash_epsilon,
        ).to(self.device)
        self.critic = ContinuousPPOCritic(
            self.state_dimension, self.critic_hidden_dimensions
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
            raise ValueError(f"states must have shape [batch, {self.state_dimension}].")
        if np.any(~np.isfinite(array)):
            raise ValueError("states must be finite.")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def act_xiaowan(
        self,
        states: np.ndarray | Sequence[float],
        xiaowan_geometry: np.ndarray,
        *,
        deterministic: bool,
        xiaowan_active: np.ndarray | Sequence[bool] | None = None,
    ) -> dict[str, np.ndarray]:
        tensor = self._state_tensor(states)
        geometry = np.asarray(xiaowan_geometry, dtype=np.float32)
        if geometry.shape != (len(tensor), XIAOWAN_GEOMETRY_DIMENSION):
            raise ValueError(
                "xiaowan_geometry must have shape [batch, 3]."
            )
        geometry_tensor = torch.as_tensor(
            geometry, dtype=torch.float32, device=self.device
        )
        active_tensor: torch.Tensor | None = None
        if xiaowan_active is not None:
            active = np.asarray(xiaowan_active, dtype=np.bool_)
            if active.shape != (len(tensor),):
                raise ValueError("xiaowan_active must have shape [batch].")
            active_tensor = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            decision = self.actor.act_xiaowan(
                tensor,
                geometry_tensor,
                deterministic=deterministic,
                xiaowan_active=active_tensor,
            )
        return {
            name: decision[name].cpu().numpy().astype(
                np.float32, copy=False
            )
            for name in ("latent", "fraction", "mean", "std", "log_prob")
        }

    def complete_action(
        self,
        states: np.ndarray | Sequence[float],
        xiaowan_decision: Mapping[str, np.ndarray],
        xiaowan_geometry: np.ndarray,
        nuozhadu_geometry: np.ndarray,
        *,
        deterministic: bool,
        nuozhadu_active: np.ndarray | Sequence[bool] | None = None,
    ) -> dict[str, np.ndarray]:
        tensor = self._state_tensor(states)
        xw_geometry = np.asarray(xiaowan_geometry, dtype=np.float32)
        nzd_geometry = np.asarray(nuozhadu_geometry, dtype=np.float32)
        if xw_geometry.shape != (len(tensor), XIAOWAN_GEOMETRY_DIMENSION):
            raise ValueError("xiaowan_geometry must have shape [batch, 3].")
        if nzd_geometry.shape != (
            len(tensor), NUOZHADU_GEOMETRY_DIMENSION
        ):
            raise ValueError("nuozhadu_geometry must have shape [batch, 4].")
        latent = np.asarray(xiaowan_decision["latent"], dtype=np.float32)
        if latent.shape != (len(tensor),):
            raise ValueError("XiaoWan latent action must have shape [batch].")
        active_tensor: torch.Tensor | None = None
        if nuozhadu_active is not None:
            active = np.asarray(nuozhadu_active, dtype=np.bool_)
            if active.shape != (len(tensor),):
                raise ValueError("nuozhadu_active must have shape [batch].")
            active_tensor = torch.as_tensor(
                active, dtype=torch.bool, device=self.device
            )
        with torch.no_grad():
            decision = self.actor.complete_action(
                tensor,
                torch.as_tensor(latent, dtype=torch.float32, device=self.device),
                torch.as_tensor(
                    xw_geometry, dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    nzd_geometry, dtype=torch.float32, device=self.device
                ),
                deterministic=deterministic,
                nuozhadu_active=active_tensor,
            )
            values = self.critic(tensor)
        return {
            "latent_actions": decision["latent_actions"].cpu().numpy().astype(np.float32, copy=False),
            "action_fractions": decision["action_fractions"].cpu().numpy().astype(np.float32, copy=False),
            "log_probs": decision["log_probs"].cpu().numpy().astype(np.float32, copy=False),
            "component_log_probs": decision["component_log_probs"].cpu().numpy().astype(np.float32, copy=False),
            "values": values.cpu().numpy().astype(np.float32, copy=False),
            "sample_entropies": decision["sample_entropies"].cpu().numpy().astype(np.float32, copy=False),
            "action_std": decision["action_std"].cpu().numpy().astype(np.float32, copy=False),
            "action_geometry": np.concatenate(
                (xw_geometry, nzd_geometry), axis=1
            ).astype(np.float32, copy=False),
        }

    def predict_values(self, states: np.ndarray | Sequence[float]) -> np.ndarray:
        tensor = self._state_tensor(states)
        with torch.no_grad():
            values = self.critic(tensor)
        return values.cpu().numpy().astype(np.float32, copy=False)

    def set_learning_rate(self, learning_rate: float) -> None:
        value = _finite_float("learning_rate", learning_rate)
        if value < 0.0:
            raise ValueError("learning_rate must be nonnegative.")
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def update(self, rollout: ContinuousPPORolloutBatch) -> dict[str, float | int]:
        n_steps, num_envs = rollout.validate(state_dimension=self.state_dimension)
        rollout_size = n_steps * num_envs
        if rollout_size % self.minibatch_size != 0:
            raise ValueError("rollout size must be divisible by minibatch_size.")
        advantages, returns = compute_continuous_gae(
            rollout, gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        flat_states = np.asarray(rollout.states, dtype=np.float32).reshape(
            rollout_size, self.state_dimension
        )
        flat_latent = np.asarray(rollout.latent_actions, dtype=np.float32).reshape(
            rollout_size, CONTINUOUS_ACTION_DIMENSION
        )
        flat_old_log_probs = np.asarray(rollout.old_log_probs, dtype=np.float32).reshape(rollout_size)
        flat_active = np.asarray(
            rollout.active_action_dimensions, dtype=np.bool_
        ).reshape(rollout_size, CONTINUOUS_ACTION_DIMENSION)
        flat_geometry = np.asarray(
            rollout.action_geometry, dtype=np.float32
        ).reshape(rollout_size, ACTION_GEOMETRY_DIMENSION)
        flat_old_values = np.asarray(rollout.values, dtype=np.float32).reshape(rollout_size)
        flat_advantages = advantages.reshape(rollout_size)
        flat_returns = returns.reshape(rollout_size)
        advantage_mean_before = float(flat_advantages.mean())
        advantage_std_before = float(flat_advantages.std(ddof=0))
        if self.advantage_normalization:
            flat_advantages = (
                flat_advantages - advantage_mean_before
            ) / max(advantage_std_before, 1.0e-8)

        states = torch.as_tensor(flat_states, dtype=torch.float32, device=self.device)
        latent_actions = torch.as_tensor(flat_latent, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(flat_old_log_probs, dtype=torch.float32, device=self.device)
        active_dimensions = torch.as_tensor(
            flat_active, dtype=torch.float32, device=self.device
        )
        action_geometry = torch.as_tensor(
            flat_geometry, dtype=torch.float32, device=self.device
        )
        normalized_advantages = torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device)
        return_targets = torch.as_tensor(flat_returns, dtype=torch.float32, device=self.device)

        metric_names = (
            "policy_loss", "value_loss", "policy_entropy", "total_loss",
            "approx_kl", "clip_fraction", "ratio_mean", "ratio_std",
            "gradient_norm", "actor_gradient_norm", "critic_gradient_norm",
            "xiaowan_action_std", "nuozhadu_action_std",
        )
        totals = {name: 0.0 for name in metric_names}
        updates = 0
        ratio_max = 0.0
        actor_parameters = list(self.actor.parameters())
        critic_parameters = list(self.critic.parameters())
        all_parameters = [*actor_parameters, *critic_parameters]

        self.actor.train()
        self.critic.train()
        epochs_completed = 0
        early_stop_triggered = False
        last_epoch_approx_kl = 0.0
        for _ in range(self.update_epochs):
            epoch_kl_total = 0.0
            epoch_updates = 0
            indices = torch.randperm(rollout_size, device="cpu")
            for start in range(0, rollout_size, self.minibatch_size):
                batch = indices[start : start + self.minibatch_size].to(self.device)
                evaluated = self.actor.evaluate_latent_actions(
                    states[batch],
                    latent_actions[batch],
                    action_geometry[batch],
                )
                components = evaluated["component_log_probs"]
                new_log_probs = (
                    components * active_dimensions[batch]
                ).sum(dim=1)
                new_values = self.critic(states[batch])
                entropy = (
                    -components * active_dimensions[batch]
                ).sum(dim=1).mean()
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * normalized_advantages[batch]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * normalized_advantages[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.mean((new_values - return_targets[batch]) ** 2)
                total_loss = (
                    policy_loss
                    + self.value_loss_coefficient * value_loss
                    - self.entropy_coefficient * entropy
                )
                if not torch.isfinite(total_loss):
                    raise FloatingPointError("Continuous PPO produced non-finite loss.")

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                actor_gradient_norm = _gradient_norm(actor_parameters)
                critic_gradient_norm = _gradient_norm(critic_parameters)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        all_parameters, self.max_gradient_norm
                    ).detach().item()
                )
                if not np.isfinite(gradient_norm):
                    raise FloatingPointError("Continuous PPO produced non-finite gradient norm.")
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_range)
                        .to(dtype=torch.float32)
                        .mean()
                    )
                    ratio_std = ratio.std(unbiased=False)
                    ratio_max = max(ratio_max, float(ratio.max().item()))
                    std = evaluated["action_std"].mean(dim=0)
                current = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "policy_entropy": entropy,
                    "total_loss": total_loss,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "ratio_mean": ratio.mean(),
                    "ratio_std": ratio_std,
                    "gradient_norm": torch.as_tensor(gradient_norm),
                    "actor_gradient_norm": torch.as_tensor(actor_gradient_norm),
                    "critic_gradient_norm": torch.as_tensor(critic_gradient_norm),
                    "xiaowan_action_std": std[0],
                    "nuozhadu_action_std": std[1],
                }
                for name, value in current.items():
                    totals[name] += float(value.detach().item())
                epoch_kl_total += float(approx_kl.detach().item())
                epoch_updates += 1
                updates += 1

            epochs_completed += 1
            if epoch_updates:
                last_epoch_approx_kl = epoch_kl_total / epoch_updates
            if last_epoch_approx_kl > self.target_kl:
                early_stop_triggered = True
                break

        if updates == 0:
            raise RuntimeError("Continuous PPO update performed no optimizer steps.")
        metrics: dict[str, float | int] = {
            name: value / updates for name, value in totals.items()
        }
        metrics.update(
            {
                "ratio_max": ratio_max,
                "advantage_mean_before_normalization": advantage_mean_before,
                "advantage_std_before_normalization": advantage_std_before,
                "return_mean": float(flat_returns.mean()),
                "return_std": float(flat_returns.std(ddof=0)),
                "old_value_prediction_mean": float(flat_old_values.mean()),
                "old_value_prediction_std": float(flat_old_values.std(ddof=0)),
                "rollout_size": rollout_size,
                "minibatch_updates": updates,
                "epochs_completed": epochs_completed,
                "early_stop_triggered": early_stop_triggered,
                "last_epoch_approx_kl": last_epoch_approx_kl,
                "target_kl": self.target_kl,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
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
            "action_dimension": CONTINUOUS_ACTION_DIMENSION,
            "action_mode": CONTINUOUS_ACTION_MODE,
            "action_geometry_dimension": ACTION_GEOMETRY_DIMENSION,
            "xiaowan_geometry_dimension": XIAOWAN_GEOMETRY_DIMENSION,
            "nuozhadu_geometry_dimension": NUOZHADU_GEOMETRY_DIMENSION,
            "actor_hidden_dimensions": list(self.actor_hidden_dimensions),
            "critic_hidden_dimensions": list(self.critic_hidden_dimensions),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "entropy_coefficient": self.entropy_coefficient,
            "value_loss_coefficient": self.value_loss_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "target_kl": self.target_kl,
            "advantage_normalization": self.advantage_normalization,
            "adam_epsilon": self.adam_epsilon,
            "initial_log_std": self.initial_log_std,
            "min_log_std": self.min_log_std,
            "max_log_std": self.max_log_std,
            "squash_epsilon": self.squash_epsilon,
            "seed": self.seed,
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
                raise ValueError("Checkpoint CUDA RNG count does not match devices.")
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(item, device="cpu") for item in cuda_states]
            )

    def training_state(self) -> dict[str, Any]:
        return {
            "schema_version": CONTINUOUS_PPO_AGENT_STATE_SCHEMA_VERSION,
            "configuration": self.configuration_state(),
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "rng_state": self.rng_state(),
        }

    def _validate_state_configuration(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != CONTINUOUS_PPO_AGENT_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported continuous PPO agent-state schema.")
        configuration = state.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("Continuous PPO state is missing configuration.")
        for name, value in self.configuration_state().items():
            if configuration.get(name) != value:
                raise ValueError(f"Checkpoint continuous PPO {name} differs.")

    def load_training_state(self, state: Mapping[str, Any], *, restore_rng: bool) -> None:
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
    "AutoregressiveSquashedGaussianActor",
    "CONTINUOUS_ACTION_DIMENSION",
    "CONTINUOUS_ACTION_MODE",
    "ContinuousPPOAgent",
    "ContinuousPPOCritic",
    "ContinuousPPORolloutBatch",
    "compute_continuous_gae",
    "set_continuous_ppo_global_seed",
]
