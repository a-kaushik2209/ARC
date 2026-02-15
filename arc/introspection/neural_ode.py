# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ODEFunc(nn.Module):

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 16,
    ):
        super().__init__()

        self.state_dim = state_dim

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(state_dim + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0], 1)
        elif t.dim() == 1 and t.shape[0] != z.shape[0]:
            t = t.unsqueeze(1).expand(z.shape[0], 1)
        else:
            t = t.view(-1, 1)

        t_emb = self.time_embed(t)

        z_t = torch.cat([z, t_emb], dim=-1)
        dzdt = self.net(z_t)

        return dzdt

class NeuralODESolver:
    def __init__(self, method: str = "rk4"):
        self.method = method

    def solve(
        self,
        func: ODEFunc,
        z0: torch.Tensor,
        t_span: Tuple[float, float],
        n_steps: int = 10,
    ) -> torch.Tensor:
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps

        z = z0
        t = torch.tensor(t0, dtype=z0.dtype, device=z0.device)

        for _ in range(n_steps):
            if self.method == "euler":
                z = z + dt * func(t, z)

            elif self.method == "rk4":
                dt_tensor = torch.tensor(dt, dtype=z.dtype, device=z.device)

                k1 = func(t, z)
                k2 = func(t + dt_tensor/2, z + dt_tensor/2 * k1)
                k3 = func(t + dt_tensor/2, z + dt_tensor/2 * k2)
                k4 = func(t + dt_tensor, z + dt_tensor * k3)

                z = z + dt_tensor/6 * (k1 + 2*k2 + 2*k3 + k4)

            t = t + dt

        return z

    def solve_trajectory(
        self,
        func: ODEFunc,
        z0: torch.Tensor,
        t_eval: torch.Tensor,
    ) -> torch.Tensor:
        trajectory = [z0]
        z = z0

        for i in range(len(t_eval) - 1):
            z = self.solve(
                func, z,
                t_span=(t_eval[i].item(), t_eval[i+1].item()),
                n_steps=5,
            )
            trajectory.append(z)

        return torch.stack(trajectory)

class TrainingDynamicsODE(nn.Module):
    def __init__(
        self,
        signal_dim: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.signal_dim = signal_dim

        self.ode_func = ODEFunc(
            state_dim=signal_dim,
            hidden_dim=hidden_dim,
        )

        self.solver = NeuralODESolver(method="rk4")

        self._state_history: deque = deque(maxlen=100)
        self._time_history: deque = deque(maxlen=100)

    def record_state(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        weight_norm: float,
        epoch: int,
        **kwargs
    ) -> None:
        state = torch.tensor([
            loss,
            gradient_norm,
            learning_rate,
            weight_norm,
            math.log(epoch + 1),
            math.sin(epoch * 0.1),
            math.cos(epoch * 0.1),
        ] + [0.0] * (self.signal_dim - 7), dtype=torch.float32)

        self._state_history.append(state)
        self._time_history.append(float(epoch))

    def predict_trajectory(
        self,
        n_steps_ahead: int = 10,
    ) -> torch.Tensor:
        if len(self._state_history) < 2:
            return torch.zeros(n_steps_ahead, self.signal_dim)

        z0 = self._state_history[-1].unsqueeze(0)
        t_current = self._time_history[-1]

        t_eval = torch.linspace(
            t_current,
            t_current + n_steps_ahead,
            n_steps_ahead + 1,
        )

        with torch.no_grad():
            trajectory = self.solver.solve_trajectory(
                self.ode_func, z0, t_eval
            )

        return trajectory[1:, 0, :]

    def compute_loss(self) -> torch.Tensor:
        if len(self._state_history) < 10:
            return torch.tensor(0.0)

        states = torch.stack(list(self._state_history))
        times = torch.tensor(list(self._time_history))

        mid = len(states) // 2

        z0 = states[:mid].mean(dim=0, keepdim=True)
        t_span = (times[0].item(), times[-1].item())

        z_pred = self.solver.solve(
            self.ode_func, z0, t_span, n_steps=mid
        )

        z_actual = states[mid:].mean(dim=0, keepdim=True)

        loss = F.mse_loss(z_pred, z_actual)

        return loss

@dataclass
class PredictedTrajectory:
    future_losses: List[float]
    future_gradient_norms: List[float]
    convergence_epoch: Optional[int]
    divergence_risk: float
    time_to_failure: Optional[float]

class TrajectoryPredictor:

    def __init__(self, signal_dim: int = 20):
        self.dynamics = TrainingDynamicsODE(signal_dim=signal_dim)
        self._training_active = False

    def update(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        weight_norm: float,
        epoch: int,
    ) -> None:
        self.dynamics.record_state(
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=learning_rate,
            weight_norm=weight_norm,
            epoch=epoch,
        )

    def predict(self, n_epochs: int = 20) -> PredictedTrajectory:
        trajectory = self.dynamics.predict_trajectory(n_epochs)

        future_losses = trajectory[:, 0].tolist()
        future_gradients = trajectory[:, 1].tolist()

        convergence_epoch = None
        for i, loss in enumerate(future_losses):
            if loss < 0.01:
                convergence_epoch = i
                break

        if future_losses:
            trend = np.polyfit(range(len(future_losses)), future_losses, 1)[0]
            divergence_risk = min(1.0, max(0.0, trend * 10))
        else:
            divergence_risk = 0.5

        return PredictedTrajectory(
            future_losses=future_losses,
            future_gradient_norms=future_gradients,
            convergence_epoch=convergence_epoch,
            divergence_risk=divergence_risk,
            time_to_failure=None if divergence_risk < 0.5 else 5.0,
        )

class ContinuousLearningRateController:
    def __init__(
        self,
        initial_lr: float = 0.001,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
    ):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        self._velocity_history: deque = deque(maxlen=20)

    def update(
        self,
        dynamics_velocity: float,
        lyapunov_exponent: float,
    ) -> float:
        self._velocity_history.append(dynamics_velocity)

        if lyapunov_exponent > 0.1:
            self.current_lr *= 0.9
        elif lyapunov_exponent < -0.1 and dynamics_velocity < 0.01:
            self.current_lr *= 1.1

        self.current_lr = max(self.min_lr, min(self.max_lr, self.current_lr))

        return self.current_lr