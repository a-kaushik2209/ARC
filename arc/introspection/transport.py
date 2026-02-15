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
import numpy as np
import torch
import torch.nn as nn
from collections import deque

@dataclass
class TransportState:
    wasserstein_distance: float
    cumulative_transport: float
    mode_collapse_risk: float
    drift_velocity: float
    distribution_entropy: float
    barycenter_shift: float

class SinkhornDistance:
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iters: int = 100,
        threshold: float = 1e-3,
    ):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.threshold = threshold

    def compute(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        n = source.shape[0]
        m = target.shape[0]

        a = torch.ones(n, device=source.device) / n
        b = torch.ones(m, device=target.device) / m

        C = torch.cdist(source, target, p=2) ** 2

        K = torch.exp(-C / self.epsilon)

        u = torch.ones(n, device=source.device)

        for _ in range(self.max_iters):
            u_prev = u.clone()

            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)

            if torch.max(torch.abs(u - u_prev)).item() < self.threshold:
                break

        v = b / (K.T @ u + 1e-10)
        transport_cost = torch.sum(u.unsqueeze(1) * K * v.unsqueeze(0) * C)

        return float(transport_cost.sqrt().item())

    def transport_plan(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        n = source.shape[0]
        m = target.shape[0]

        a = torch.ones(n, device=source.device) / n
        b = torch.ones(m, device=target.device) / m

        C = torch.cdist(source, target, p=2) ** 2
        K = torch.exp(-C / self.epsilon)

        u = torch.ones(n, device=source.device)

        for _ in range(self.max_iters):
            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)

        v = b / (K.T @ u + 1e-10)
        plan = u.unsqueeze(1) * K * v.unsqueeze(0)

        return plan

class DistributionTracker:

    def __init__(
        self,
        n_samples: int = 1000,
        n_bins: int = 50,
    ):
        self.n_samples = n_samples
        self.n_bins = n_bins

        self._distributions: deque = deque(maxlen=20)

        self._barycenters: deque = deque(maxlen=50)

        self._stats_history: deque = deque(maxlen=100)

    def update(self, activations: torch.Tensor) -> Dict[str, float]:
        flat = activations.detach().flatten()

        if len(flat) > self.n_samples:
            indices = torch.randperm(len(flat))[:self.n_samples]
            samples = flat[indices]
        else:
            samples = flat

        stats = {
            "mean": samples.mean().item(),
            "std": samples.std().item(),
            "min": samples.min().item(),
            "max": samples.max().item(),
            "sparsity": (samples.abs() < 0.01).float().mean().item(),
        }

        hist, _ = np.histogram(samples.cpu().numpy(), bins=self.n_bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        stats["entropy"] = -np.sum(hist * np.log(hist))

        self._distributions.append(samples.cpu())
        self._barycenters.append(samples.mean().item())
        self._stats_history.append(stats)

        return stats

    def get_distribution_drift(self) -> float:
        if len(self._distributions) < 2:
            return 0.0

        prev = self._distributions[-2].unsqueeze(1)
        curr = self._distributions[-1].unsqueeze(1)

        sinkhorn = SinkhornDistance()
        return sinkhorn.compute(prev, curr)

    def mode_collapse_risk(self) -> float:
        if not self._stats_history:
            return 0.0

        recent = list(self._stats_history)[-5:]

        mean_std = np.mean([s["std"] for s in recent])
        std_risk = 1.0 / (1.0 + mean_std)

        mean_entropy = np.mean([s["entropy"] for s in recent])
        entropy_risk = 1.0 / (1.0 + mean_entropy)

        mean_sparsity = np.mean([s["sparsity"] for s in recent])
        sparsity_risk = mean_sparsity

        return 0.4 * std_risk + 0.4 * entropy_risk + 0.2 * sparsity_risk

    def barycenter_velocity(self) -> float:
        if len(self._barycenters) < 2:
            return 0.0

        bary = np.array(list(self._barycenters))
        return float(np.abs(np.diff(bary)).mean())

class WassersteinMonitor:

    def __init__(self, model: nn.Module):
        self.model = model
        self.sinkhorn = SinkhornDistance()
        self.tracker = DistributionTracker()

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_activations: Dict[str, torch.Tensor] = {}

        self._transport_history: deque = deque(maxlen=100)
        self._cumulative_transport: float = 0.0

        self._attach_hooks()

    def _attach_hooks(self) -> None:
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._captured_activations[name] = output.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.Linear)):
                handle = module.register_forward_hook(make_hook(name))
                self._hooks.append(handle)

    def update(self) -> TransportState:
        if not self._captured_activations:
            return TransportState(
                wasserstein_distance=0,
                cumulative_transport=0,
                mode_collapse_risk=0,
                drift_velocity=0,
                distribution_entropy=0,
                barycenter_shift=0,
            )

        all_activations = torch.cat([
            act.flatten()[:1000]
            for act in self._captured_activations.values()
        ])

        stats = self.tracker.update(all_activations)

        w_dist = self.tracker.get_distribution_drift()
        self._transport_history.append(w_dist)
        self._cumulative_transport += w_dist

        if len(self._transport_history) >= 3:
            drift_velocity = np.mean(list(self._transport_history)[-5:])
        else:
            drift_velocity = 0.0

        self._captured_activations.clear()

        return TransportState(
            wasserstein_distance=w_dist,
            cumulative_transport=self._cumulative_transport,
            mode_collapse_risk=self.tracker.mode_collapse_risk(),
            drift_velocity=drift_velocity,
            distribution_entropy=stats.get("entropy", 0),
            barycenter_shift=self.tracker.barycenter_velocity(),
        )

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()