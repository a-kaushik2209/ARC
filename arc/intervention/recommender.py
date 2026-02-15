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
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import numpy as np
import json

from arc.config import FailureMode
from arc.intervention.actions import (
    InterventionAction,
    ActionParameters,
    InterventionResult,
    DEFAULT_INTERVENTIONS,
)

@dataclass
class Recommendation:
    action: InterventionAction
    parameters: ActionParameters
    confidence: float
    rationale: str
    alternatives: List[InterventionAction] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    estimated_effectiveness: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.name,
            "action_description": self.action.description,
            "parameters": self.parameters.to_dict(),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "alternatives": [a.name for a in self.alternatives],
            "caveats": self.caveats,
            "estimated_effectiveness": self.estimated_effectiveness,
        }

class EffectivenessTracker:
    def __init__(self):
        self._outcomes: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"successes": 0, "total": 0}
        )

        self._priors: Dict[str, float] = {
            InterventionAction.REDUCE_LR.name: 0.7,
            InterventionAction.ENABLE_GRAD_CLIPPING.name: 0.8,
            InterventionAction.EARLY_STOP.name: 0.9,
            InterventionAction.INCREASE_WEIGHT_DECAY.name: 0.6,
        }
        self._default_prior = 0.5

    def record_outcome(self, result: InterventionResult) -> None:
        key = (result.action.name, result.failure_mode or "UNKNOWN")

        if result.was_effective is not None:
            self._outcomes[key]["total"] += 1
            if result.was_effective:
                self._outcomes[key]["successes"] += 1

    def get_effectiveness(
        self,
        action: InterventionAction,
        failure_mode: FailureMode
    ) -> Tuple[float, float]:
        key = (action.name, failure_mode.name)
        counts = self._outcomes.get(key, {"successes": 0, "total": 0})

        prior = self._priors.get(action.name, self._default_prior)
        alpha = prior * k + counts["successes"]
        beta = (1 - prior) * k + counts["total"] - counts["successes"]

        effectiveness = alpha / (alpha + beta)

        confidence = min(1.0, counts["total"] / 10)

        return effectiveness, confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcomes": dict(self._outcomes),
            "priors": self._priors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EffectivenessTracker':
        tracker = cls()
        tracker._outcomes = defaultdict(
            lambda: {"successes": 0, "total": 0},
            {tuple(k.split("_")): v for k, v in d.get("outcomes", {}).items()}
        )
        tracker._priors = d.get("priors", {})
        return tracker

class InterventionRecommender:
    def __init__(self):
        self.effectiveness_tracker = EffectivenessTracker()
        self._intervention_history: List[InterventionResult] = []

    def recommend(
        self,
        failure_mode: FailureMode,
        failure_probability: float,
        contributing_signals: Optional[List[str]] = None,
        current_lr: float = 0.001,
        current_epoch: int = 0,
    ) -> Recommendation:
        if failure_probability < 0.3:
            return Recommendation(
                action=InterventionAction.NO_ACTION,
                parameters=ActionParameters(),
                confidence=0.9,
                rationale="Training appears healthy with low failure risk.",
                caveats=["Continue monitoring; predictions are probabilistic."],
                estimated_effectiveness=1.0,
            )
        if failure_probability < 0.5:
            return Recommendation(
                action=InterventionAction.CONTINUE_MONITORING,
                parameters=ActionParameters(),
                confidence=0.7,
                rationale=f"Moderate {failure_mode} risk detected. Close monitoring advised.",
                caveats=[
                    "Risk may increase if trend continues.",
                    "Prepare interventions but don't apply yet.",
                ],
                estimated_effectiveness=0.8,
            )

        candidates = DEFAULT_INTERVENTIONS.get(
            failure_mode.name,
            [InterventionAction.CONTINUE_MONITORING]
        )

        ranked_actions = []
        for action in candidates:
            eff, conf = self.effectiveness_tracker.get_effectiveness(
                action, failure_mode
            )
            score = eff * (0.5 + 0.5 * conf)
            ranked_actions.append((action, eff, conf, score))

        ranked_actions.sort(key=lambda x: x[3], reverse=True)

        best_action, best_eff, best_conf, _ = ranked_actions[0]

        parameters = self._generate_parameters(
            best_action,
            failure_mode,
            failure_probability,
            contributing_signals,
            current_lr,
        )

        rationale = self._generate_rationale(
            best_action,
            failure_mode,
            failure_probability,
            contributing_signals,
        )

        caveats = self._generate_caveats(best_action, failure_mode)

        confidence = min(0.9, failure_probability * best_conf)

        return Recommendation(
            action=best_action,
            parameters=parameters,
            confidence=confidence,
            rationale=rationale,
            alternatives=[a for a, _, _, _ in ranked_actions[1:3]],
            caveats=caveats,
            estimated_effectiveness=best_eff,
        )

    def _generate_parameters(
        self,
        action: InterventionAction,
        failure_mode: FailureMode,
        probability: float,
        signals: Optional[List[str]],
        current_lr: float,
    ) -> ActionParameters:
        params = ActionParameters()

        if action == InterventionAction.REDUCE_LR:
            factor = 0.5 if probability > 0.8 else 0.7
            params.lr_factor = factor
            params.parameter_confidence["lr_factor"] = 0.8

        elif action == InterventionAction.INCREASE_LR:
            factor = 2.0 if probability > 0.8 else 1.5
            params.lr_factor = factor
            params.parameter_confidence["lr_factor"] = 0.6

        elif action == InterventionAction.ENABLE_GRAD_CLIPPING:
            params.clip_max_norm = 1.0
            params.parameter_confidence["clip_max_norm"] = 0.7

        elif action == InterventionAction.INCREASE_WEIGHT_DECAY:
            params.weight_decay = 0.01 if probability > 0.7 else 0.001
            params.parameter_confidence["weight_decay"] = 0.6

        elif action == InterventionAction.ADD_DROPOUT:
            params.dropout_rate = 0.3 if probability > 0.8 else 0.2
            params.parameter_confidence["dropout_rate"] = 0.5

        elif action == InterventionAction.EARLY_STOP:
            params.patience = 5
            params.parameter_confidence["patience"] = 0.8

        return params

    def _generate_rationale(
        self,
        action: InterventionAction,
        failure_mode: FailureMode,
        probability: float,
        signals: Optional[List[str]],
    ) -> str:
        mode_str = str(failure_mode).lower()
        signal_str = ""
        if signals:
            signal_str = f" Key signals: {', '.join(signals[:3])}."

        rationales = {
            InterventionAction.REDUCE_LR: (
                f"High {mode_str} risk ({probability:.0%}) detected. "
                f"Reducing learning rate should stabilize optimization.{signal_str}"
            ),
            InterventionAction.INCREASE_LR: (
                f"{mode_str.capitalize()} risk ({probability:.0%}) suggests insufficient learning. "
                f"Increasing learning rate may help escape plateau.{signal_str}"
            ),
            InterventionAction.ENABLE_GRAD_CLIPPING: (
                f"Gradient-related issues detected ({probability:.0%} risk). "
                f"Clipping prevents explosive updates.{signal_str}"
            ),
            InterventionAction.INCREASE_WEIGHT_DECAY: (
                f"{mode_str.capitalize()} risk ({probability:.0%}) may benefit from regularization. "
                f"Weight decay encourages simpler solutions.{signal_str}"
            ),
            InterventionAction.EARLY_STOP: (
                f"High {mode_str} risk ({probability:.0%}). "
                f"Stopping now preserves best checkpoint.{signal_str}"
            ),
        }

        return rationales.get(
            action,
            f"{action.description} recommended due to {mode_str} risk ({probability:.0%}).{signal_str}"
        )

    def _generate_caveats(
        self,
        action: InterventionAction,
        failure_mode: FailureMode,
    ) -> List[str]:
        caveats = [
            "Intervention effectiveness varies by architecture and dataset.",
            "Predictions are probabilistic; false positives may occur.",
        ]

        action_caveats = {
            InterventionAction.REDUCE_LR: [
                "May slow convergence significantly.",
                "Consider warmup if reducing by more than 50%.",
            ],
            InterventionAction.INCREASE_LR: [
                "Risk of destabilizing training.",
                "Monitor closely after application.",
            ],
            InterventionAction.EARLY_STOP: [
                "May leave performance on the table.",
                "Consider other interventions first.",
            ],
            InterventionAction.ADD_DROPOUT: [
                "Requires architecture modification.",
                "May require retraining from scratch.",
            ],
        }

        caveats.extend(action_caveats.get(action, []))

        return caveats

    def record_outcome(self, result: InterventionResult) -> None:
        self._intervention_history.append(result)
        self.effectiveness_tracker.record_outcome(result)

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._intervention_history]

    def save(self, path: str) -> None:
        data = {
            "effectiveness": self.effectiveness_tracker.to_dict(),
            "history": self.get_intervention_history(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'InterventionRecommender':
        with open(path, 'r') as f:
            data = json.load(f)

        recommender = cls()
        recommender.effectiveness_tracker = EffectivenessTracker.from_dict(
            data.get("effectiveness", {})
        )
        return recommender