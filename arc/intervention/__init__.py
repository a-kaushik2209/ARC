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

from arc.intervention.actions import (
    InterventionAction,
    ActionParameters,
    InterventionResult,
)
from arc.intervention.recommender import InterventionRecommender
from arc.intervention.rollback import (
    WeightRollback,
    RollbackConfig,
    RollbackAction,
    create_self_healing_arc,
)
from arc.intervention.amp import (
    AMPRollback,
    AMPConfig,
    SafeAutocast,
    create_amp_training,
)
from arc.intervention.heuristics import (
    DiffusionHeuristics,
    LLMHeuristics,
    DetectionHeuristics,
    AutoHeuristics,
)
from arc.intervention.oom_handler import (
    OOMRecoveryHandler,
    OOMConfig,
    OOMRecoveryFailed,
    OOMStrategy,
    oom_protected,
    get_memory_stats,
    estimate_model_memory,
)
from arc.intervention.hardware_handler import (
    HardwareRecoveryHandler,
    HardwareConfig,
    HardwareErrorType,
    hardware_safe,
    get_best_device,
    get_device_info,
)

__all__ = [
    "InterventionAction",
    "ActionParameters",
    "InterventionResult",
    "InterventionRecommender",
    "WeightRollback",
    "RollbackConfig",
    "RollbackAction",
    "create_self_healing_arc",
    "AMPRollback",
    "AMPConfig",
    "SafeAutocast",
    "create_amp_training",
    "DiffusionHeuristics",
    "LLMHeuristics",
    "DetectionHeuristics",
    "AutoHeuristics",
]