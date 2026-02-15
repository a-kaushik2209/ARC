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
from dataclasses import dataclass, field
import numpy as np
import json

@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    @property
    def range(self) -> float:
        if self.count == 0:
            return 0.0
        return self.max_val - self.min_val

    def normalize_zscore(self, value: float, epsilon: float = 1e-8) -> float:
        if self.count < 2:
            return 0.0
        std = self.std
        if std < epsilon:
            return 0.0
        return (value - self.mean) / std

    def normalize_minmax(self, value: float, epsilon: float = 1e-8) -> float:
        if self.count == 0:
            return 0.0
        r = self.range
        if r < epsilon:
            return 0.5
        return (value - self.min_val) / r

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "M2": self.M2,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunningStats':
        return cls(
            count=d["count"],
            mean=d["mean"],
            M2=d["M2"],
            min_val=d["min_val"],
            max_val=d["max_val"],
        )

class RobustRunningStats:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self._samples: List[float] = []
        self._sorted = False

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return

        self._samples.append(value)
        self._sorted = False

        if len(self._samples) > self.buffer_size:
            np.random.shuffle(self._samples)
            self._samples = self._samples[:self.buffer_size // 2]

    def _ensure_sorted(self) -> None:
        if not self._sorted:
            self._samples.sort()
            self._sorted = True

    @property
    def median(self) -> float:
        if not self._samples:
            return 0.0
        self._ensure_sorted()
        return self._samples[len(self._samples) // 2]

    @property
    def iqr(self) -> float:
        if len(self._samples) < 4:
            return 0.0
        self._ensure_sorted()
        n = len(self._samples)
        q1 = self._samples[n // 4]
        q3 = self._samples[3 * n // 4]
        return q3 - q1

    def normalize_robust(self, value: float, epsilon: float = 1e-8) -> float:
        iqr = self.iqr
        if iqr < epsilon:
            return 0.0
        return (value - self.median) / (iqr * 0.7413)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer_size": self.buffer_size,
            "samples": self._samples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RobustRunningStats':
        stats = cls(buffer_size=d["buffer_size"])
        stats._samples = d["samples"]
        return stats

class OnlineNormalizer:

    def __init__(
        self,
        method: str = "zscore",
        robust: bool = False,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
    ):
        self.method = method
        self.robust = robust
        self.clip_range = clip_range

        self._stats: Dict[str, RunningStats] = {}
        self._robust_stats: Dict[str, RobustRunningStats] = {}

        self._feature_order: List[str] = []

    def update(self, features: Dict[str, float]) -> None:
        for name, value in features.items():
            if not np.isfinite(value):
                continue

            if name not in self._stats:
                self._stats[name] = RunningStats()
                self._feature_order.append(name)
            self._stats[name].update(value)

            if self.robust or self.method == "robust":
                if name not in self._robust_stats:
                    self._robust_stats[name] = RobustRunningStats()
                self._robust_stats[name].update(value)

    def normalize(
        self,
        features: Dict[str, float],
        return_array: bool = False
    ) -> Any:
        normalized = {}

        for name, value in features.items():
            if name not in self._stats:
                normalized[name] = np.clip(value, *self.clip_range)
                continue

            if self.method == "zscore":
                if self.robust and name in self._robust_stats:
                    norm_val = self._robust_stats[name].normalize_robust(value)
                else:
                    norm_val = self._stats[name].normalize_zscore(value)
            elif self.method == "minmax":
                norm_val = self._stats[name].normalize_minmax(value)
            elif self.method == "robust":
                if name in self._robust_stats:
                    norm_val = self._robust_stats[name].normalize_robust(value)
                else:
                    norm_val = self._stats[name].normalize_zscore(value)
            else:
                norm_val = value

            normalized[name] = float(np.clip(norm_val, *self.clip_range))

        if return_array:
            return np.array([
                normalized.get(name, 0.0) for name in self._feature_order
            ])

        return normalized

    def normalize_batch(
        self,
        feature_batch: List[Dict[str, float]]
    ) -> np.ndarray:
        return np.vstack([
            self.normalize(f, return_array=True) for f in feature_batch
        ])

    @property
    def n_features(self) -> int:
        return len(self._feature_order)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_order.copy()

    def get_statistics(self, feature_name: str) -> Optional[Dict[str, float]]:
        if feature_name not in self._stats:
            return None

        stats = self._stats[feature_name]
        result = {
            "count": stats.count,
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min_val,
            "max": stats.max_val,
        }

        if feature_name in self._robust_stats:
            robust = self._robust_stats[feature_name]
            result["median"] = robust.median
            result["iqr"] = robust.iqr

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "robust": self.robust,
            "clip_range": list(self.clip_range),
            "feature_order": self._feature_order,
            "stats": {name: s.to_dict() for name, s in self._stats.items()},
            "robust_stats": {name: s.to_dict() for name, s in self._robust_stats.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OnlineNormalizer':
        normalizer = cls(
            method=d["method"],
            robust=d["robust"],
            clip_range=tuple(d["clip_range"]),
        )
        normalizer._feature_order = d["feature_order"]
        normalizer._stats = {
            name: RunningStats.from_dict(s) for name, s in d["stats"].items()
        }
        normalizer._robust_stats = {
            name: RobustRunningStats.from_dict(s) for name, s in d.get("robust_stats", {}).items()
        }
        return normalizer

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'OnlineNormalizer':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))