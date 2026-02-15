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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.signals.base import SignalCollector, SignalSnapshot
from arc.config import SignalConfig

class SharpnessCollector(SignalCollector):

    def __init__(
        self,
        config: SignalConfig,
        num_hutchinson_samples: int = 5,
        power_iterations: int = 10,
        perturbation_radius: float = 0.05,
    ):

        super().__init__(config)
        self.num_hutchinson_samples = num_hutchinson_samples
        self.power_iterations = power_iterations
        self.perturbation_radius = perturbation_radius

        self._hessian_trace: float = 0.0
        self._max_eigenvalue: float = 0.0
        self._gradient_curvature: float = 0.0
        self._sam_sharpness: float = 0.0
        self._flatness_ratio: float = 0.0

        self._trace_ema: float = 0.0
        self._max_eig_ema: float = 0.0
        self._ema_alpha: float = 0.1

    @property
    def name(self) -> str:
        return "sharpness"

    def compute_sharpness(
        self,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:

        model = model or self._model
        if model is None:
            return {}

        self._hessian_trace = self._hutchinson_trace(
            model, loss_fn, inputs, targets
        )

        self._max_eigenvalue = self._power_iteration_max_eig(
            model, loss_fn, inputs, targets
        )

        self._gradient_curvature = self._gradient_direction_curvature(
            model, loss_fn, inputs, targets
        )

        self._sam_sharpness = self._sam_perturbation_sharpness(
            model, loss_fn, inputs, targets
        )

        if self._max_eigenvalue > 1e-10:
            self._flatness_ratio = self._hessian_trace / self._max_eigenvalue

        self._trace_ema = self._ema_alpha * self._hessian_trace + (1 - self._ema_alpha) * self._trace_ema
        self._max_eig_ema = self._ema_alpha * self._max_eigenvalue + (1 - self._ema_alpha) * self._max_eig_ema

        return {
            "hessian_trace": self._hessian_trace,
            "max_eigenvalue": self._max_eigenvalue,
            "gradient_curvature": self._gradient_curvature,
            "sam_sharpness": self._sam_sharpness,
            "flatness_ratio": self._flatness_ratio,
        }

    def _hutchinson_trace(
        self,
        model: nn.Module,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:

        model.eval()
        traces = []

        for _ in range(self.num_hutchinson_samples):

            v = []
            for p in model.parameters():
                if p.requires_grad:
                    v.append(torch.randint_like(p, 0, 2) * 2.0 - 1.0)

            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            grads = torch.autograd.grad(
                loss,
                [p for p in model.parameters() if p.requires_grad],
                create_graph=True,
                allow_unused=True,
            )

            gv = sum(
                (g * vi).sum()
                for g, vi in zip(grads, v)
                if g is not None
            )

            hvp = torch.autograd.grad(
                gv,
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=False,
                allow_unused=True,
            )

            vhv = sum(
                (vi * hv).sum().item()
                for vi, hv in zip(v, hvp)
                if hv is not None
            )

            traces.append(vhv)

        model.train()
        return float(np.mean(traces))

    def _power_iteration_max_eig(
        self,
        model: nn.Module,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:

        model.eval()

        v = []
        for p in model.parameters():
            if p.requires_grad:
                vi = torch.randn_like(p)
                vi = vi / (vi.norm() + 1e-10)
                v.append(vi)

        for _ in range(self.power_iterations):

            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            grads = torch.autograd.grad(
                loss,
                [p for p in model.parameters() if p.requires_grad],
                create_graph=True,
                allow_unused=True,
            )

            gv = sum(
                (g * vi).sum()
                for g, vi in zip(grads, v)
                if g is not None
            )

            hvp = torch.autograd.grad(
                gv,
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=False,
                allow_unused=True,
            )

            hvp_list = [hv if hv is not None else torch.zeros_like(vi) for hv, vi in zip(hvp, v)]
            total_norm = sum((hv ** 2).sum() for hv in hvp_list).sqrt()

            if total_norm > 1e-10:
                v = [hv / total_norm for hv in hvp_list]

        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=True,
            allow_unused=True,
        )

        gv = sum(
            (g * vi).sum()
            for g, vi in zip(grads, v)
            if g is not None
        )

        hvp = torch.autograd.grad(
            gv,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=False,
            allow_unused=True,
        )

        eigenvalue = sum(
            (vi * hv).sum().item()
            for vi, hv in zip(v, hvp)
            if hv is not None
        )

        model.train()
        return float(max(0, eigenvalue))

    def _gradient_direction_curvature(
        self,
        model: nn.Module,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:

        model.eval()
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=True,
            allow_unused=True,
        )

        grad_norm_sq = sum((g ** 2).sum() for g in grads if g is not None)

        if grad_norm_sq < 1e-10:
            model.train()
            return 0.0

        gg = sum((g ** 2).sum() for g in grads if g is not None)

        hvp = torch.autograd.grad(
            gg,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=False,
            allow_unused=True,
        )

        curvature = 0.5 * sum(
            (g * hv).sum().item()
            for g, hv in zip(grads, hvp)
            if g is not None and hv is not None
        ) / grad_norm_sq.item()

        model.train()
        return float(curvature)

    def _sam_perturbation_sharpness(
        self,
        model: nn.Module,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:

        model.eval()

        model.zero_grad()
        outputs = model(inputs)
        loss_original = loss_fn(outputs, targets)
        loss_original.backward()

        original_params = {}
        grad_norm = 0.0

        for name, p in model.named_parameters():
            if p.grad is not None:
                original_params[name] = p.data.clone()
                grad_norm += (p.grad ** 2).sum().item()

        grad_norm = np.sqrt(grad_norm) + 1e-10

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.data += self.perturbation_radius * p.grad / grad_norm

        with torch.no_grad():
            outputs_perturbed = model(inputs)
            loss_perturbed = loss_fn(outputs_perturbed, targets)

        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in original_params:
                    p.data = original_params[name]

        sharpness = loss_perturbed.item() - loss_original.item()

        model.train()
        return float(max(0, sharpness))

    def collect(self) -> SignalSnapshot:

        signals = {
            "sharpness": {
                "hessian_trace": self._hessian_trace,
                "hessian_trace_ema": self._trace_ema,
                "max_eigenvalue": self._max_eigenvalue,
                "max_eigenvalue_ema": self._max_eig_ema,
                "gradient_curvature": self._gradient_curvature,
                "sam_sharpness": self._sam_sharpness,
                "flatness_ratio": self._flatness_ratio,
            }
        }

        return SignalSnapshot(signals=signals, overhead_ms=0)

    def reset(self) -> None:

        super().reset()
        self._hessian_trace = 0.0
        self._max_eigenvalue = 0.0
        self._gradient_curvature = 0.0
        self._sam_sharpness = 0.0
        self._flatness_ratio = 0.0