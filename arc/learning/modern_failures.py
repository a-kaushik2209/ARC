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

from arc.config import FailureMode

@dataclass
class GrokingSignals:
    train_val_divergence: float
    divergence_duration: int
    weight_norm_velocity: float
    representation_rank_trend: float
    in_memorization_phase: bool
    grokking_probability: float

@dataclass
class DoubleDescentSignals:
    interpolation_proximity: float
    test_loss_non_monotonicity: float
    model_complexity_ratio: float
    in_critical_regime: bool
    double_descent_probability: float

class GrokkingDetector:
    def __init__(
        self,
        min_memorization_epochs: int = 20,
        train_loss_threshold: float = 0.01,
        gap_threshold: float = 0.5,
    ):
        self.min_memorization_epochs = min_memorization_epochs
        self.train_loss_threshold = train_loss_threshold
        self.gap_threshold = gap_threshold

        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._weight_norms: List[float] = []
        self._representation_ranks: List[float] = []

        self._memorization_start_epoch: Optional[int] = None
        self._in_memorization: bool = False

    def update(
        self,
        train_loss: float,
        val_loss: float,
        weight_norm: Optional[float] = None,
        representation_rank: Optional[float] = None,
    ) -> GrokingSignals:
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)

        if weight_norm is not None:
            self._weight_norms.append(weight_norm)
        if representation_rank is not None:
            self._representation_ranks.append(representation_rank)

        is_memorizing = train_loss < self.train_loss_threshold
        gap = val_loss - train_loss

        if is_memorizing and gap > self.gap_threshold:
            if not self._in_memorization:
                self._in_memorization = True
                self._memorization_start_epoch = len(self._train_losses) - 1
        else:
            self._in_memorization = False
            self._memorization_start_epoch = None

        divergence_duration = 0
        if self._memorization_start_epoch is not None:
            divergence_duration = len(self._train_losses) - self._memorization_start_epoch

        weight_norm_velocity = 0.0
        if len(self._weight_norms) >= 3:
            recent = self._weight_norms[-5:]
            weight_norm_velocity = (recent[-1] - recent[0]) / len(recent)

        rank_trend = 0.0
        if len(self._representation_ranks) >= 3:
            recent = self._representation_ranks[-5:]
            rank_trend = np.polyfit(range(len(recent)), recent, 1)[0]

        probability = self._compute_grokking_probability(
            gap, divergence_duration, weight_norm_velocity, rank_trend
        )

        return GrokingSignals(
            train_val_divergence=gap,
            divergence_duration=divergence_duration,
            weight_norm_velocity=weight_norm_velocity,
            representation_rank_trend=rank_trend,
            in_memorization_phase=self._in_memorization,
            grokking_probability=probability,
        )

    def _compute_grokking_probability(
        self,
        gap: float,
        duration: int,
        weight_velocity: float,
        rank_trend: float,
    ) -> float:
        score = 0.0

        if gap > self.gap_threshold:
            score += min(0.3, gap * 0.3)

        if duration > self.min_memorization_epochs:
            score += min(0.3, duration / 100)

        if weight_velocity > 0:
            score += min(0.2, weight_velocity * 10)

        if rank_trend > 0:
            score += min(0.2, rank_trend * 5)

        return min(1.0, score)

    def reset(self) -> None:
        self._train_losses.clear()
        self._val_losses.clear()
        self._weight_norms.clear()
        self._representation_ranks.clear()
        self._in_memorization = False
        self._memorization_start_epoch = None

class DoubleDescentDetector:

    def __init__(
        self,
        critical_ratio_range: Tuple[float, float] = (0.8, 1.2),
        smoothing_window: int = 5,
    ):
        self.critical_ratio_range = critical_ratio_range
        self.smoothing_window = smoothing_window

        self._test_losses: List[float] = []
        self._complexity_ratios: List[float] = []

    def update(
        self,
        test_loss: float,
        model_params: int,
        train_samples: int,
    ) -> DoubleDescentSignals:
        complexity_ratio = model_params / max(train_samples, 1)

        self._test_losses.append(test_loss)
        self._complexity_ratios.append(complexity_ratio)

        in_critical = (
            self.critical_ratio_range[0] <= complexity_ratio <= self.critical_ratio_range[1]
        )

        proximity = 1.0 - abs(complexity_ratio - 1.0)
        proximity = max(0, min(1, proximity))

        non_monotonicity = self._compute_non_monotonicity()

        probability = self._compute_double_descent_probability(
            in_critical, proximity, non_monotonicity
        )

        return DoubleDescentSignals(
            interpolation_proximity=proximity,
            test_loss_non_monotonicity=non_monotonicity,
            model_complexity_ratio=complexity_ratio,
            in_critical_regime=in_critical,
            double_descent_probability=probability,
        )

    def _compute_non_monotonicity(self) -> float:
        if len(self._test_losses) < self.smoothing_window:
            return 0.0

        losses = np.array(self._test_losses)
        smoothed = np.convolve(losses, np.ones(self.smoothing_window) / self.smoothing_window, mode='valid')

        if len(smoothed) < 3:
            return 0.0

        diffs = np.diff(smoothed)
        signs = np.sign(diffs)
        changes = np.sum(np.abs(np.diff(signs)) == 2)

        return changes / len(diffs)

    def _compute_double_descent_probability(
        self,
        in_critical: bool,
        proximity: float,
        non_monotonicity: float,
    ) -> float:
        score = 0.0

        if in_critical:
            score += 0.4

        score += proximity * 0.3

        score += non_monotonicity * 0.3

        return min(1.0, score)

    def reset(self) -> None:
        self._test_losses.clear()
        self._complexity_ratios.clear()

class ModernFailureDetector:

    def __init__(self):
        self.grokking = GrokkingDetector()
        self.double_descent = DoubleDescentDetector()

    def update(
        self,
        train_loss: float,
        val_loss: float,
        model_params: int = 0,
        train_samples: int = 0,
        weight_norm: Optional[float] = None,
        representation_rank: Optional[float] = None,
    ) -> Dict[str, Any]:
        grokking = self.grokking.update(
            train_loss, val_loss, weight_norm, representation_rank
        )

        double_descent = None
        if model_params > 0 and train_samples > 0:
            double_descent = self.double_descent.update(
                val_loss, model_params, train_samples
            )

        return {
            "grokking": grokking,
            "double_descent": double_descent,
            "highest_risk_mode": self._get_highest_risk(grokking, double_descent),
        }

    def _get_highest_risk(
        self,
        grokking: GrokingSignals,
        double_descent: Optional[DoubleDescentSignals],
    ) -> Optional[FailureMode]:
        risks = [
            (FailureMode.GROKKING_RISK, grokking.grokking_probability),
        ]

        if double_descent:
            risks.append(
                (FailureMode.DOUBLE_DESCENT, double_descent.double_descent_probability)
            )

        highest = max(risks, key=lambda x: x[1])
        if highest[1] > 0.5:
            return highest[0]

        return None

    def reset(self) -> None:
        self.grokking.reset()
        self.double_descent.reset()