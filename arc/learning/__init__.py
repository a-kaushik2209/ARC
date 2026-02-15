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

from arc.learning.simulator import FailureSimulator, SimulatedTrajectory
from arc.learning.labeler import TrajectoryLabeler, FailureLabel
from arc.learning.meta_model import TrainingDynamicsPredictor
from arc.learning.trainer import MetaModelTrainer

from arc.learning.mamba_model import (
    MambaPredictor,
    MambaBlock,
    SelectiveSSM,
    MultiScaleTemporalFusion,
    SignalCrossAttention,
)
from arc.learning.modern_failures import (
    GrokkingDetector,
    DoubleDescentDetector,
    ModernFailureDetector,
    GrokingSignals,
    DoubleDescentSignals,
)

__all__ = [
    "FailureSimulator",
    "SimulatedTrajectory",
    "TrajectoryLabeler",
    "FailureLabel",
    "TrainingDynamicsPredictor",
    "MetaModelTrainer",
    "MambaPredictor",
    "MambaBlock",
    "SelectiveSSM",
    "MultiScaleTemporalFusion",
    "SignalCrossAttention",
    "GrokkingDetector",
    "DoubleDescentDetector",
    "ModernFailureDetector",
    "GrokingSignals",
    "DoubleDescentSignals",
]