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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import json

class FailureMode(Enum):

    DIVERGENCE = auto()
    VANISHING_GRADIENTS = auto()
    EXPLODING_GRADIENTS = auto()
    REPRESENTATION_COLLAPSE = auto()
    SEVERE_OVERFITTING = auto()

    GROKKING_RISK = auto()
    DOUBLE_DESCENT = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

@dataclass
class SignalConfig:

    collect_every_n_steps: int = 1
    aggregate_per_epoch: bool = True

    activation_sample_ratio: float = 0.1
    activation_sample_layers: Optional[List[str]] = None

    compute_gradient_entropy: bool = True
    gradient_histogram_bins: int = 50

    compute_curvature_proxy: bool = False
    curvature_hvp_samples: int = 5

    track_effective_rank: bool = True
    effective_rank_sample_size: int = 100

@dataclass
class FeatureConfig:

    window_size: int = 10
    min_history: int = 3

    compute_trend: bool = True
    compute_curvature: bool = True
    compute_spectral: bool = False

    compute_correlations: bool = True
    correlation_pairs: Optional[List[tuple]] = None

@dataclass
class PredictionConfig:

    temporal_hidden_size: int = 64
    num_gru_layers: int = 2
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    cnn_kernel_size: int = 3
    dropout_rate: float = 0.2

    mc_dropout_samples: int = 20
    confidence_level: float = 0.95

    temperature_scaling: bool = True

    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4

@dataclass
class FailureThresholds:

    loss_explosion_factor: float = 10.0
    loss_nan_detection: bool = True

    vanishing_grad_threshold: float = 1e-7
    vanishing_grad_epochs: int = 5

    exploding_grad_threshold: float = 1e4
    exploding_grad_epochs: int = 2

    activation_similarity_threshold: float = 0.95
    effective_rank_collapse_ratio: float = 0.3

    overfit_gap_threshold: float = 0.5
    overfit_val_plateau_epochs: int = 10

@dataclass
class OverheadConfig:

    max_overhead_percent: float = 5.0

    adaptive_sampling: bool = True
    overhead_check_interval: int = 100

    reduce_activation_sampling: bool = True
    disable_curvature_proxy: bool = True
    reduce_mc_samples: bool = True

@dataclass
class Config:

    signal: SignalConfig = field(default_factory=SignalConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    thresholds: FailureThresholds = field(default_factory=FailureThresholds)
    overhead: OverheadConfig = field(default_factory=OverheadConfig)

    verbose: bool = False
    log_level: str = "WARNING"

    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:

        def _asdict_recursive(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _asdict_recursive(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.name
            elif isinstance(obj, list):
                return [_asdict_recursive(i) for i in obj]
            else:
                return obj
        return _asdict_recursive(self)

    def to_json(self, indent: int = 2) -> str:

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':

        return cls(
            signal=SignalConfig(**d.get('signal', {})),
            feature=FeatureConfig(**d.get('feature', {})),
            prediction=PredictionConfig(**d.get('prediction', {})),
            thresholds=FailureThresholds(**d.get('thresholds', {})),
            overhead=OverheadConfig(**d.get('overhead', {})),
            verbose=d.get('verbose', False),
            log_level=d.get('log_level', 'WARNING'),
            device=d.get('device', 'auto'),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'Config':

        return cls.from_dict(json.loads(json_str))

    @classmethod
    def low_overhead(cls) -> 'Config':

        config = cls()
        config.signal.activation_sample_ratio = 0.05
        config.signal.compute_curvature_proxy = False
        config.signal.compute_gradient_entropy = False
        config.feature.compute_spectral = False
        config.feature.compute_correlations = False
        config.prediction.mc_dropout_samples = 5
        config.overhead.max_overhead_percent = 2.0
        return config

    @classmethod
    def high_accuracy(cls) -> 'Config':

        config = cls()
        config.signal.activation_sample_ratio = 0.3
        config.signal.compute_curvature_proxy = True
        config.feature.compute_spectral = True
        config.feature.window_size = 20
        config.prediction.mc_dropout_samples = 50
        config.overhead.max_overhead_percent = 10.0
        return config
