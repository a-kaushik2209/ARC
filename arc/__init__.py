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

__version__ = "2.0.0"
__author__ = "Arc Research Team"

from arc.config import Config, FailureMode

from arc.api.callback import Arc
from arc.api.pytorch_callback import ArcCallback, ArcWrapper
from arc.api.report import ReportGenerator

from arc.api.v2 import ArcV2, Arc2

from arc.signals import (
    SignalCollector,
    CompositeCollector,
    GradientCollector,
    ActivationCollector,
    WeightCollector,
    OptimizerCollector,
    LossCollector,
    CurvatureCollector,
)

from arc.features import (
    SignalBuffer,
    EpochSnapshot,
    FeatureExtractor,
    OnlineNormalizer,
)

from arc.prediction import (
    FailurePredictor,
    FailurePrediction,
    SignalAttribution,
    AttributionEngine,
    ProbabilityCalibrator,
)

from arc.intervention import (
    InterventionAction,
    InterventionRecommender,
)

from arc.evaluation import (
    EvaluationMetrics,
    BenchmarkSuite,
    BenchmarkResult,
)

from arc.learning import (
    FailureSimulator,
    TrajectoryLabeler,
    TrainingDynamicsPredictor,
    MetaModelTrainer,
)

from arc.learning.ewc import ElasticWeightConsolidation, ProgressiveNet

from arc.uncertainty import (
    ConformalPredictor,
    ConformalRegression,
    VennAbersCalibrator,
)

from arc.security import (
    AdversarialDetector,
    AdversarialTrainer,
    RandomizedSmoothing,
)

from arc.physics import (
    PINNStabilizer,
    AdaptiveLossBalancer,
    CurriculumScheduler,
)

from arc.features.spectral import (
    FourierFeatureEncoder,
    SpectralAnalyzer,
    MultiScaleEncoder,
)

from arc.api.bulletproof import (
    BulletproofTrainer,
    BulletproofConfig,
    protect,
    protect_trainer,
)
from arc.intervention.oom_handler import (
    OOMRecoveryHandler,
    OOMConfig,
    OOMRecoveryFailed,
    get_memory_stats,
    estimate_model_memory,
)
from arc.signals.silent_detector import (
    SilentCrashDetector,
    SilentDetectorConfig,
    SilentFailureType,
)
from arc.intervention.hardware_handler import (
    HardwareRecoveryHandler,
    HardwareConfig,
    get_best_device,
    get_device_info,
)

__all__ = [

    "Arc",
    "ArcV2",
    "Arc2",
    "ArcCallback",
    "ArcWrapper",
    "ReportGenerator",
    "Config",
    "FailureMode",

    "SignalCollector",
    "CompositeCollector",
    "GradientCollector",
    "ActivationCollector",
    "WeightCollector",
    "OptimizerCollector",
    "LossCollector",
    "CurvatureCollector",

    "SignalBuffer",
    "EpochSnapshot",
    "FeatureExtractor",
    "OnlineNormalizer",

    "FailurePredictor",
    "FailurePrediction",
    "SignalAttribution",
    "AttributionEngine",
    "ProbabilityCalibrator",

    "InterventionAction",
    "InterventionRecommender",

    "EvaluationMetrics",
    "BenchmarkSuite",
    "BenchmarkResult",

    "FailureSimulator",
    "TrajectoryLabeler",
    "TrainingDynamicsPredictor",
    "MetaModelTrainer",

    "ElasticWeightConsolidation",
    "ProgressiveNet",

    "ConformalPredictor",
    "ConformalRegression",
    "VennAbersCalibrator",

    "AdversarialDetector",
    "AdversarialTrainer",
    "RandomizedSmoothing",

    "PINNStabilizer",
    "AdaptiveLossBalancer",
    "CurriculumScheduler",

    "FourierFeatureEncoder",
    "SpectralAnalyzer",
    "MultiScaleEncoder",

    "__version__",
]
