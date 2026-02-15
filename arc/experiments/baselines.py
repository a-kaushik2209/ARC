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

from typing import Optional, List
import numpy as np
from collections import deque

class BaselineDetector:

    def __init__(self):
        self.name = "BaselineDetector"

    def update(self, **kwargs) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        pass

class RandomDetector(BaselineDetector):

    def __init__(self, detection_probability: float = 0.1):
        super().__init__()
        self.name = "Random"
        self.detection_probability = detection_probability

    def update(self, **kwargs) -> bool:
        return np.random.random() < self.detection_probability

class GradientThresholdDetector(BaselineDetector):

    def __init__(
        self,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 1e3,
    ):
        super().__init__()
        self.name = "GradientThreshold"
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

    def update(self, gradient_norm: float = 0, **kwargs) -> bool:
        return (
            gradient_norm < self.vanishing_threshold or
            gradient_norm > self.exploding_threshold
        )

class LossCurveDetector(BaselineDetector):
    def __init__(
        self,
        window_size: int = 5,
        increase_threshold: int = 3,
    ):
        super().__init__()
        self.name = "LossCurve"
        self.window_size = window_size
        self.increase_threshold = increase_threshold
        self._losses: deque = deque(maxlen=window_size)

    def update(self, loss: float = 0, **kwargs) -> bool:
        self._losses.append(loss)

        if len(self._losses) < 3:
            return False

        increases = 0
        for i in range(1, len(self._losses)):
            if self._losses[i] > self._losses[i-1]:
                increases += 1

        return increases >= self.increase_threshold

    def reset(self) -> None:
        self._losses.clear()

class LossValueDetector(BaselineDetector):

    def __init__(self, threshold: float = 10.0):
        super().__init__()
        self.name = "LossValue"
        self.threshold = threshold

    def update(self, loss: float = 0, **kwargs) -> bool:
        return loss > self.threshold or np.isnan(loss)

class EnsembleDetector(BaselineDetector):

    def __init__(self):
        super().__init__()
        self.name = "Ensemble"
        self.detectors = [
            GradientThresholdDetector(),
            LossCurveDetector(),
            LossValueDetector(),
        ]

    def update(self, **kwargs) -> bool:
        votes = sum(d.update(**kwargs) for d in self.detectors)
        return votes >= 2

    def reset(self) -> None:
        for d in self.detectors:
            d.reset()

class AdaptiveThresholdDetector(BaselineDetector):

    def __init__(self, z_threshold: float = 3.0, min_samples: int = 10):
        super().__init__()
        self.name = "AdaptiveThreshold"
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self._loss_history: List[float] = []
        self._grad_history: List[float] = []

    def update(self, loss: float = 0, gradient_norm: float = 0, **kwargs) -> bool:
        self._loss_history.append(loss)
        self._grad_history.append(gradient_norm)

        if len(self._loss_history) < self.min_samples:
            return False

        loss_mean = np.mean(self._loss_history[:-1])
        loss_std = np.std(self._loss_history[:-1]) + 1e-10
        loss_z = abs(loss - loss_mean) / loss_std

        grad_mean = np.mean(self._grad_history[:-1])
        grad_std = np.std(self._grad_history[:-1]) + 1e-10
        grad_z = abs(gradient_norm - grad_mean) / grad_std

        return loss_z > self.z_threshold or grad_z > self.z_threshold

    def reset(self) -> None:
        self._loss_history.clear()
        self._grad_history.clear()

def get_all_baselines() -> List[BaselineDetector]:
    return [
        RandomDetector(),
        GradientThresholdDetector(),
        LossCurveDetector(),
        LossValueDetector(),
        EnsembleDetector(),
        AdaptiveThresholdDetector(),
    ]