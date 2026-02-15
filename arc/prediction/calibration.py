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

class ProbabilityCalibrator:

    def __init__(self, method: str = "temperature"):

        self.method = method
        self.temperature: float = 1.0
        self.platt_params: Optional[Tuple[float, float]] = None
        self.isotonic_bins: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        if self.method == "temperature":
            self._fit_temperature(logits, labels)
        elif self.method == "platt":
            self._fit_platt(logits, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(logits, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._is_fitted = True

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:

        if not self._is_fitted:

            return F.softmax(logits, dim=-1)

        if self.method == "temperature":
            return F.softmax(logits / self.temperature, dim=-1)
        elif self.method == "platt":
            return self._apply_platt(logits)
        elif self.method == "isotonic":
            return self._apply_isotonic(logits)
        else:
            return F.softmax(logits, dim=-1)

    def _fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        temperature = nn.Parameter(torch.ones(1) * 1.0)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

        def eval_temp():
            optimizer.zero_grad()

            temp = torch.clamp(temperature, min=0.1)
            scaled_logits = logits / temp
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_temp)

        self.temperature = max(0.1, temperature.item())

    def _fit_platt(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        max_logits = logits.max(dim=-1).values
        targets = (logits.argmax(dim=-1) == labels).float()

        a = nn.Parameter(torch.ones(1))
        b = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100)

        def eval_platt():
            optimizer.zero_grad()
            probs = torch.sigmoid(a * max_logits + b)
            loss = F.binary_cross_entropy(probs, targets)
            loss.backward()
            return loss

        optimizer.step(eval_platt)

        self.platt_params = (a.item(), b.item())

    def _apply_platt(self, logits: torch.Tensor) -> torch.Tensor:

        if self.platt_params is None:
            return F.softmax(logits, dim=-1)

        a, b = self.platt_params

        scaled = a * logits + b
        return F.softmax(scaled, dim=-1)

    def _fit_isotonic(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:

        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values.detach().numpy()
        correctness = (logits.argmax(dim=-1) == labels).float().numpy()

        sorted_indices = np.argsort(max_probs)
        sorted_probs = max_probs[sorted_indices]
        sorted_correct = correctness[sorted_indices]

        n = len(sorted_probs)
        calibrated = np.zeros(n)

        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_frequencies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n):
            bin_idx = min(int(sorted_probs[i] * n_bins), n_bins - 1)
            bin_frequencies[bin_idx] += sorted_correct[i]
            bin_counts[bin_idx] += 1

        bin_accuracies = np.where(
            bin_counts > 0,
            bin_frequencies / bin_counts,
            bin_boundaries[:-1] + 0.5 / n_bins
        )

        for i in range(1, n_bins):
            if bin_accuracies[i] < bin_accuracies[i-1]:

                combined = (
                    bin_frequencies[i-1] + bin_frequencies[i]
                ) / max(bin_counts[i-1] + bin_counts[i], 1)
                bin_accuracies[i-1] = combined
                bin_accuracies[i] = combined

        self.isotonic_bins = (bin_boundaries, bin_accuracies)

    def _apply_isotonic(self, logits: torch.Tensor) -> torch.Tensor:

        if self.isotonic_bins is None:
            return F.softmax(logits, dim=-1)

        bin_boundaries, bin_accuracies = self.isotonic_bins
        n_bins = len(bin_accuracies)

        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        calibrated_max = torch.zeros_like(max_probs)
        for i in range(len(max_probs)):
            p = max_probs[i].item()
            bin_idx = min(int(p * n_bins), n_bins - 1)
            calibrated_max[i] = bin_accuracies[bin_idx]

        scale = calibrated_max / (max_probs + 1e-10)
        calibrated_probs = probs * scale.unsqueeze(-1)

        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=-1, keepdim=True)

        return calibrated_probs

    def expected_calibration_error(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:

        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = in_bin.float().mean()

            if in_bin.any():
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

        return ece.item()

    def calibration_curve(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
    ) -> Dict[str, list]:

        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == labels).float()

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        mean_predicted = []
        fraction_positive = []
        bin_sizes = []

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])

            if in_bin.any():
                mean_predicted.append(confidences[in_bin].mean().item())
                fraction_positive.append(accuracies[in_bin].mean().item())
                bin_sizes.append(in_bin.sum().item())
            else:
                mean_predicted.append((bin_boundaries[i] + bin_boundaries[i+1]).item() / 2)
                fraction_positive.append(0.0)
                bin_sizes.append(0)

        return {
            "mean_predicted_value": mean_predicted,
            "fraction_of_positives": fraction_positive,
            "bin_sizes": bin_sizes,
        }

    def to_dict(self) -> Dict[str, Any]:

        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt_params": self.platt_params,
            "isotonic_bins": (
                self.isotonic_bins[0].tolist() if self.isotonic_bins else None,
                self.isotonic_bins[1].tolist() if self.isotonic_bins else None,
            ),
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProbabilityCalibrator':

        calibrator = cls(method=d["method"])
        calibrator.temperature = d["temperature"]
        calibrator.platt_params = d["platt_params"]
        if d["isotonic_bins"][0] is not None:
            calibrator.isotonic_bins = (
                np.array(d["isotonic_bins"][0]),
                np.array(d["isotonic_bins"][1]),
            )
        calibrator._is_fitted = d["is_fitted"]
        return calibrator
