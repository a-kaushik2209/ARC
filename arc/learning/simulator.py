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

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import copy
import numpy as np
import torch
import torch.nn as nn

from arc.config import FailureMode

@dataclass
class SimulatedTrajectory:
    signals: List[Dict[str, Any]]
    failure_mode: Optional[FailureMode]
    failure_epoch: Optional[int]
    severity: float
    parameters: Dict[str, Any]

    @property
    def is_failure(self) -> bool:
        return self.failure_mode is not None

    @property
    def n_epochs(self) -> int:
        return len(self.signals)

class FailureSimulator:
    def __init__(
        self,
        device: str = "cpu",
        base_lr: float = 0.001,
        max_epochs: int = 50,
        seed: Optional[int] = None,
    ):
        self.device = device
        self.base_lr = base_lr
        self.max_epochs = max_epochs

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_divergence_trajectory(
        self,
        severity: float = 0.7,
        n_epochs: int = 30,
    ) -> SimulatedTrajectory:
        signals = []

        explosion_epoch = int(n_epochs * (1 - severity * 0.5))
        explosion_rate = 1.5 + severity * 2.0

        loss = np.random.uniform(1.0, 3.0)
        grad_norm = np.random.uniform(0.1, 1.0)

        for epoch in range(n_epochs):
            if epoch < explosion_epoch:
                loss *= np.random.uniform(0.95, 0.99)
                grad_norm *= np.random.uniform(0.98, 1.02)
                loss_gradient = np.random.uniform(-0.1, 0)
            else:
                t = (epoch - explosion_epoch) / max(n_epochs - explosion_epoch, 1)
                loss *= explosion_rate ** t
                grad_norm *= (1.2 + severity) ** t
                loss_gradient = loss * 0.1 * (t + 1)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=loss,
                val_loss=loss * np.random.uniform(1.0, 1.1),
                grad_norm=grad_norm,
                grad_entropy=0.5 * (1 - min(t if epoch >= explosion_epoch else 0, 1)),
                activation_mean=np.random.uniform(-1, 1),
                activation_std=1.0 + (t if epoch >= explosion_epoch else 0) * 2,
                weight_update_ratio=0.01 * (explosion_rate ** (t if epoch >= explosion_epoch else 0)),
                effective_lr=self.base_lr * (3.0 if epoch >= explosion_epoch else 1.0),
                loss_gradient=loss_gradient,
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=FailureMode.DIVERGENCE,
            failure_epoch=explosion_epoch,
            severity=severity,
            parameters={"explosion_rate": explosion_rate},
        )

    def generate_vanishing_gradient_trajectory(
        self,
        severity: float = 0.7,
        n_epochs: int = 40,
    ) -> SimulatedTrajectory:
        signals = []

        vanish_start = int(n_epochs * 0.2)
        vanish_rate = 0.8 - severity * 0.15

        loss = np.random.uniform(1.0, 2.0)
        grad_norm = np.random.uniform(0.5, 1.0)
        plateau_loss = loss * np.random.uniform(0.3, 0.5)

        for epoch in range(n_epochs):
            if epoch < vanish_start:
                loss *= np.random.uniform(0.92, 0.98)
                grad_norm *= np.random.uniform(0.95, 1.05)
            else:
                t = (epoch - vanish_start) / max(n_epochs - vanish_start, 1)
                grad_norm *= vanish_rate
                loss = loss * 0.99 + plateau_loss * 0.01
                grad_norm = max(grad_norm, 1e-10)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=loss,
                val_loss=loss * np.random.uniform(1.0, 1.05),
                grad_norm=grad_norm,
                grad_entropy=0.3 + 0.5 * (1 - min(t if epoch >= vanish_start else 0, 1)),
                grad_flow_ratio=vanish_rate ** (epoch - vanish_start if epoch >= vanish_start else 0),
                activation_mean=np.random.uniform(-0.1, 0.1),
                activation_std=0.5 * (vanish_rate ** (epoch / 10)),
                weight_update_ratio=0.01 * (vanish_rate ** (epoch if epoch >= vanish_start else 0)),
                effective_lr=self.base_lr,
                loss_gradient=-0.01 if epoch < vanish_start else -0.001,
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=FailureMode.VANISHING_GRADIENTS,
            failure_epoch=vanish_start,
            severity=severity,
            parameters={"vanish_rate": vanish_rate},
        )

    def generate_exploding_gradient_trajectory(
        self,
        severity: float = 0.7,
        n_epochs: int = 25,
    ) -> SimulatedTrajectory:
        signals = []

        explode_epoch = int(n_epochs * (0.5 - severity * 0.3))
        explosion_factor = 2.0 + severity * 3.0

        loss = np.random.uniform(1.0, 2.0)
        grad_norm = np.random.uniform(0.1, 0.5)

        for epoch in range(n_epochs):
            if epoch < explode_epoch:
                loss *= np.random.uniform(0.95, 0.98)
                grad_norm *= np.random.uniform(0.9, 1.1)
            else:
                t = (epoch - explode_epoch) / max(n_epochs - explode_epoch, 1)
                grad_norm *= explosion_factor ** (0.3 + t * 0.7)
                loss = loss * (1 + 0.1 * t * severity)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=loss,
                val_loss=loss * np.random.uniform(1.0, 1.1),
                grad_norm=grad_norm,
                grad_entropy=0.4,
                activation_mean=np.random.uniform(-1, 1) * (1 + t if epoch >= explode_epoch else 1),
                activation_std=1.0 + (t if epoch >= explode_epoch else 0) * 3,
                weight_update_ratio=0.01 * (explosion_factor ** (t if epoch >= explode_epoch else 0)),
                effective_lr=self.base_lr,
                loss_gradient=0.1 * severity if epoch >= explode_epoch else -0.05,
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=FailureMode.EXPLODING_GRADIENTS,
            failure_epoch=explode_epoch,
            severity=severity,
            parameters={"explosion_factor": explosion_factor},
        )

    def generate_representation_collapse_trajectory(
        self,
        severity: float = 0.7,
        n_epochs: int = 50,
    ) -> SimulatedTrajectory:
        signals = []

        collapse_start = int(n_epochs * 0.3)
        similarity_target = 0.90 + severity * 0.09

        loss = np.random.uniform(1.0, 2.0)
        similarity = np.random.uniform(0.1, 0.3)
        effective_rank = np.random.uniform(50, 100)
        initial_rank = effective_rank

        for epoch in range(n_epochs):
            if epoch < collapse_start:
                loss *= np.random.uniform(0.92, 0.98)
                similarity += np.random.uniform(-0.02, 0.02)
            else:
                t = (epoch - collapse_start) / max(n_epochs - collapse_start, 1)
                similarity = similarity + (similarity_target - similarity) * 0.1
                effective_rank = initial_rank * (1 - t * severity * 0.8)
                loss *= np.random.uniform(0.99, 1.01)

            similarity = np.clip(similarity, 0, 1)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=loss,
                val_loss=loss * np.random.uniform(1.0, 1.1),
                grad_norm=np.random.uniform(0.1, 0.3),
                activation_similarity=similarity,
                effective_rank=effective_rank,
                activation_std=1.0 * (1 - t * 0.5 if epoch >= collapse_start else 1),
                dead_neuron_ratio=t * 0.3 if epoch >= collapse_start else 0.05,
                weight_update_ratio=0.005,
                effective_lr=self.base_lr,
                loss_gradient=-0.01 if epoch < collapse_start else -0.001,
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=FailureMode.REPRESENTATION_COLLAPSE,
            failure_epoch=collapse_start,
            severity=severity,
            parameters={"similarity_target": similarity_target},
        )

    def generate_overfitting_trajectory(
        self,
        severity: float = 0.7,
        n_epochs: int = 60,
    ) -> SimulatedTrajectory:
        signals = []

        overfit_start = int(n_epochs * 0.25)

        train_loss = np.random.uniform(1.0, 2.0)
        val_loss = train_loss * np.random.uniform(1.0, 1.1)
        best_val = val_loss

        for epoch in range(n_epochs):
            if epoch < overfit_start:
                train_loss *= np.random.uniform(0.92, 0.97)
                val_loss *= np.random.uniform(0.93, 0.98)
                best_val = min(best_val, val_loss)
            else:
                t = (epoch - overfit_start) / max(n_epochs - overfit_start, 1)
                train_loss *= np.random.uniform(0.95, 0.99)
                val_loss = best_val * (1 + t * severity * 0.5)

            train_val_gap = (val_loss - train_loss) / max(val_loss, 1e-8)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_val_gap=train_val_gap,
                epochs_since_improvement=epoch - overfit_start if epoch >= overfit_start else 0,
                grad_norm=np.random.uniform(0.05, 0.2),
                activation_mean=np.random.uniform(-0.5, 0.5),
                weight_update_ratio=0.01 * (0.5 ** max(0, (epoch - overfit_start) / 10)),
                effective_lr=self.base_lr,
                loss_gradient=-0.01 * (0.5 ** max(0, (epoch - overfit_start) / 10)),
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=FailureMode.SEVERE_OVERFITTING,
            failure_epoch=overfit_start,
            severity=severity,
            parameters={},
        )

    def generate_successful_trajectory(
        self,
        n_epochs: int = 50,
    ) -> SimulatedTrajectory:
        signals = []

        train_loss = np.random.uniform(1.5, 3.0)
        val_loss = train_loss * np.random.uniform(1.0, 1.05)
        grad_norm = np.random.uniform(0.3, 0.7)

        for epoch in range(n_epochs):
            decay = 0.92 + np.random.uniform(0, 0.05)
            train_loss *= decay
            val_loss *= decay * np.random.uniform(0.99, 1.01)
            grad_norm *= np.random.uniform(0.97, 1.02)
            grad_norm = np.clip(grad_norm, 0.01, 2.0)

            signals.append(self._create_signal_snapshot(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                grad_norm=grad_norm,
                grad_entropy=0.5 + np.random.uniform(-0.1, 0.1),
                activation_mean=np.random.uniform(-0.2, 0.2),
                activation_std=1.0 + np.random.uniform(-0.1, 0.1),
                activation_similarity=np.random.uniform(0.3, 0.5),
                weight_update_ratio=0.01 * decay,
                effective_lr=self.base_lr * (0.95 ** (epoch // 10)),
                loss_gradient=-train_loss * 0.05,
            ))

        return SimulatedTrajectory(
            signals=signals,
            failure_mode=None,
            failure_epoch=None,
            severity=0.0,
            parameters={},
        )

    def generate_dataset(
        self,
        n_trajectories: int = 100,
        success_ratio: float = 0.3,
        severity_range: tuple = (0.4, 0.9),
    ) -> List[SimulatedTrajectory]:
        trajectories = []

        n_success = int(n_trajectories * success_ratio)
        n_failures = n_trajectories - n_success

        for _ in range(n_success):
            n_epochs = np.random.randint(30, 70)
            trajectories.append(self.generate_successful_trajectory(n_epochs))

        failure_generators = [
            self.generate_divergence_trajectory,
            self.generate_vanishing_gradient_trajectory,
            self.generate_exploding_gradient_trajectory,
            self.generate_representation_collapse_trajectory,
            self.generate_overfitting_trajectory,
        ]

        n_per_mode = n_failures // len(failure_generators)

        for generator in failure_generators:
            for _ in range(n_per_mode):
                severity = np.random.uniform(*severity_range)
                n_epochs = np.random.randint(25, 60)
                trajectories.append(generator(severity, n_epochs))

        remaining = n_failures - n_per_mode * len(failure_generators)
        for _ in range(remaining):
            generator = np.random.choice(failure_generators)
            severity = np.random.uniform(*severity_range)
            trajectories.append(generator(severity))

        np.random.shuffle(trajectories)
        return trajectories

    def _create_signal_snapshot(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        grad_norm: float = 0.1,
        grad_entropy: float = 0.5,
        grad_flow_ratio: float = 1.0,
        activation_mean: float = 0.0,
        activation_std: float = 1.0,
        activation_similarity: float = 0.3,
        effective_rank: float = 50.0,
        dead_neuron_ratio: float = 0.05,
        weight_update_ratio: float = 0.01,
        effective_lr: float = 0.001,
        train_val_gap: Optional[float] = None,
        epochs_since_improvement: int = 0,
        loss_gradient: float = -0.01,
    ) -> Dict[str, Any]:

        if val_loss is None:
            val_loss = train_loss * np.random.uniform(1.0, 1.1)
        if train_val_gap is None:
            train_val_gap = (val_loss - train_loss) / max(val_loss, 1e-8)

        noise = lambda x, scale=0.05: x * (1 + np.random.uniform(-scale, scale))

        return {
            "epoch": epoch,
            "gradient": {
                "global": {
                    "total_grad_norm_l2": noise(grad_norm),
                    "total_grad_entropy": noise(grad_entropy),
                    "grad_flow_ratio": noise(grad_flow_ratio),
                }
            },
            "activation": {
                "global": {
                    "mean_activation_mean": noise(activation_mean, 0.1),
                    "mean_activation_std": noise(activation_std),
                    "mean_similarity": noise(activation_similarity),
                    "mean_dead_ratio": noise(dead_neuron_ratio),
                }
            },
            "weight": {
                "global": {
                    "mean_update_ratio": noise(weight_update_ratio),
                    "mean_effective_rank": noise(effective_rank),
                }
            },
            "optimizer": {
                "global": {
                    "effective_lr": effective_lr,
                }
            },
            "loss": {
                "epoch": {
                    "train_loss": noise(train_loss, 0.02),
                    "val_loss": noise(val_loss, 0.02),
                    "train_val_gap": train_val_gap,
                    "epochs_since_improvement": epochs_since_improvement,
                },
                "trajectory": {
                    "loss_gradient": noise(loss_gradient),
                }
            }
        }