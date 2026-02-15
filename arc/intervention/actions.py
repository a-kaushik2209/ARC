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
from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto

class InterventionAction(Enum):
    REDUCE_LR = auto()
    INCREASE_LR = auto()
    LR_WARMUP = auto()
    CYCLICAL_LR = auto()

    ENABLE_GRAD_CLIPPING = auto()
    ADJUST_GRAD_CLIP = auto()
    DISABLE_GRAD_CLIPPING = auto()

    INCREASE_WEIGHT_DECAY = auto()
    DECREASE_WEIGHT_DECAY = auto()
    ADD_DROPOUT = auto()
    ADD_BATCH_NORM = auto()

    INCREASE_BATCH_SIZE = auto()
    DECREASE_BATCH_SIZE = auto()

    EARLY_STOP = auto()
    CHECKPOINT_AND_ROLLBACK = auto()
    CONTINUE_MONITORING = auto()

    NO_ACTION = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def description(self) -> str:
        descriptions = {
            InterventionAction.REDUCE_LR: "Reduce learning rate to stabilize training",
            InterventionAction.INCREASE_LR: "Increase learning rate to escape local minima",
            InterventionAction.LR_WARMUP: "Apply learning rate warmup in early epochs",
            InterventionAction.CYCLICAL_LR: "Use cyclical learning rate schedule",
            InterventionAction.ENABLE_GRAD_CLIPPING: "Enable gradient clipping to prevent explosions",
            InterventionAction.ADJUST_GRAD_CLIP: "Adjust gradient clipping threshold",
            InterventionAction.DISABLE_GRAD_CLIPPING: "Disable gradient clipping",
            InterventionAction.INCREASE_WEIGHT_DECAY: "Increase weight decay for regularization",
            InterventionAction.DECREASE_WEIGHT_DECAY: "Decrease weight decay if underfitting",
            InterventionAction.ADD_DROPOUT: "Add dropout layers for regularization",
            InterventionAction.ADD_BATCH_NORM: "Add batch normalization for stability",
            InterventionAction.INCREASE_BATCH_SIZE: "Increase batch size for stable gradients",
            InterventionAction.DECREASE_BATCH_SIZE: "Decrease batch size for regularization effect",
            InterventionAction.EARLY_STOP: "Stop training to prevent further degradation",
            InterventionAction.CHECKPOINT_AND_ROLLBACK: "Rollback to last good checkpoint",
            InterventionAction.CONTINUE_MONITORING: "Continue training with increased monitoring",
            InterventionAction.NO_ACTION: "No intervention needed",
        }
        return descriptions.get(self, "Unknown action")

    @property
    def requires_restart(self) -> bool:
        restart_actions = {
            InterventionAction.ADD_DROPOUT,
            InterventionAction.ADD_BATCH_NORM,
            InterventionAction.CHECKPOINT_AND_ROLLBACK,
        }
        return self in restart_actions

    @property
    def is_reversible(self) -> bool:
        irreversible = {
            InterventionAction.EARLY_STOP,
            InterventionAction.ADD_DROPOUT,
            InterventionAction.ADD_BATCH_NORM,
        }
        return self not in irreversible

@dataclass
class ActionParameters:
    lr_factor: float = 1.0
    target_lr: Optional[float] = None

    clip_max_norm: float = 1.0
    clip_value: Optional[float] = None

    weight_decay: float = 0.01
    dropout_rate: float = 0.1

    batch_size_factor: float = 1.0

    patience: int = 5
    rollback_epochs: int = 5

    parameter_confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lr_factor": self.lr_factor,
            "target_lr": self.target_lr,
            "clip_max_norm": self.clip_max_norm,
            "clip_value": self.clip_value,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "batch_size_factor": self.batch_size_factor,
            "patience": self.patience,
            "rollback_epochs": self.rollback_epochs,
            "parameter_confidence": self.parameter_confidence,
        }

@dataclass
class InterventionResult:
    action: InterventionAction
    parameters: ActionParameters
    applied_at_epoch: int

    was_effective: Optional[bool] = None
    epochs_until_resolved: Optional[int] = None
    failure_prevented: Optional[bool] = None
    side_effects: List[str] = field(default_factory=list)

    failure_mode: Optional[str] = None
    pre_intervention_risk: float = 0.0
    post_intervention_risk: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.name,
            "parameters": self.parameters.to_dict(),
            "applied_at_epoch": self.applied_at_epoch,
            "was_effective": self.was_effective,
            "epochs_until_resolved": self.epochs_until_resolved,
            "failure_prevented": self.failure_prevented,
            "side_effects": self.side_effects,
            "failure_mode": self.failure_mode,
            "pre_intervention_risk": self.pre_intervention_risk,
            "post_intervention_risk": self.post_intervention_risk,
        }

DEFAULT_INTERVENTIONS: Dict[str, List[InterventionAction]] = {
    "DIVERGENCE": [
        InterventionAction.REDUCE_LR,
        InterventionAction.ENABLE_GRAD_CLIPPING,
        InterventionAction.CHECKPOINT_AND_ROLLBACK,
    ],
    "VANISHING_GRADIENTS": [
        InterventionAction.INCREASE_LR,
        InterventionAction.DECREASE_WEIGHT_DECAY,
        InterventionAction.LR_WARMUP,
    ],
    "EXPLODING_GRADIENTS": [
        InterventionAction.ENABLE_GRAD_CLIPPING,
        InterventionAction.REDUCE_LR,
        InterventionAction.INCREASE_BATCH_SIZE,
    ],
    "REPRESENTATION_COLLAPSE": [
        InterventionAction.INCREASE_WEIGHT_DECAY,
        InterventionAction.ADD_DROPOUT,
        InterventionAction.REDUCE_LR,
    ],
    "SEVERE_OVERFITTING": [
        InterventionAction.EARLY_STOP,
        InterventionAction.INCREASE_WEIGHT_DECAY,
        InterventionAction.ADD_DROPOUT,
    ],
}