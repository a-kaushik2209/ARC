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

from typing import Dict, Any, Optional, List
from datetime import datetime

from arc.config import FailureMode
from arc.prediction.predictor import FailurePrediction

class ReportGenerator:
    def __init__(self):
        pass

    def generate_text_report(
        self,
        prediction: FailurePrediction,
        include_history: bool = False,
        width: int = 70,
    ) -> str:
        lines = []

        lines.append("╔" + "═" * (width - 2) + "╗")
        title = f"NEURAL PROPHET - Epoch {prediction.epoch}"
        lines.append("║" + title.center(width - 2) + "║")
        lines.append("╠" + "═" * (width - 2) + "╣")

        risk_icons = {
            "low": "✅",
            "medium": "⚡",
            "high": "⚠️",
            "critical": "🚨",
            "unknown": "❓",
        }
        risk_icon = risk_icons.get(prediction.risk_level, "❓")

        lines.append("║" + " FAILURE RISK ASSESSMENT".ljust(width - 2) + "║")
        lines.append("╠" + "═" * (width - 2) + "╣")

        header = f"{'Mode':<25} {'Prob':<10} {'Confidence':<25}"
        lines.append("║ " + header + " ║")
        lines.append("╠" + "─" * (width - 2) + "╣")

        for mode in FailureMode:
            prob = prediction.failure_probabilities.get(mode, 0.0)
            ci = prediction.confidence_intervals.get(mode, (0.0, 1.0))

            prob_str = f"{prob:.2f}"

            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            if prob > 0.7:
                ci_str += " HIGH"

            mode_str = mode.name.replace("_", " ").title()
            row = f"{mode_str:<25} {prob_str:<10} {ci_str:<25}"
            lines.append("║ " + row + " ║")

        lines.append("╠" + "═" * (width - 2) + "╣")
        lines.append("║" + " TOP CONTRIBUTING SIGNALS".ljust(width - 2) + "║")
        lines.append("╠" + "─" * (width - 2) + "╣")

        for i, contrib in enumerate(prediction.top_contributors[:3], 1):
            signal_line = f" {i}. {contrib.signal_name}: {contrib.current_value:.4f} ({contrib.trend})"
            lines.append("║" + signal_line[:width-3].ljust(width - 2) + "║")

            interp_line = f"    Interpretation: {contrib.interpretation}"
            lines.append("║" + interp_line[:width-3].ljust(width - 2) + "║")

        if not prediction.top_contributors:
            lines.append("║" + "    No significant contributors identified".ljust(width - 2) + "║")

        lines.append("╠" + "═" * (width - 2) + "╣")
        lines.append("║" + " RECOMMENDED INTERVENTION".ljust(width - 2) + "║")
        lines.append("╠" + "─" * (width - 2) + "╣")

        rec = prediction.recommendation
        action_line = f" Action: {rec.action}"
        lines.append("║" + action_line[:width-3].ljust(width - 2) + "║")

        conf_line = f" Confidence: {rec.confidence:.1%}"
        lines.append("║" + conf_line[:width-3].ljust(width - 2) + "║")

        rationale = rec.rationale
        while len(rationale) > width - 5:
            lines.append("║ " + rationale[:width-4].ljust(width - 3) + "║")
            rationale = rationale[width-4:]
        lines.append("║ " + rationale.ljust(width - 3) + "║")

        if rec.caveats:
            lines.append("║" + "─" * (width - 2) + "║")
            lines.append("║ " + "Caveats:".ljust(width - 3) + "║")
            for caveat in rec.caveats[:2]:
                caveat_line = f"   • {caveat}"
                lines.append("║" + caveat_line[:width-3].ljust(width - 2) + "║")

        lines.append("╚" + "═" * (width - 2) + "╝")

        return "\n".join(lines)

    def generate_summary(self, prediction: FailurePrediction) -> str:
        mode, prob = prediction.get_highest_risk_mode()
        risk = prediction.risk_level.upper()

        if prediction.risk_level in ['high', 'critical']:
            return f"Epoch {prediction.epoch}: {risk} risk - {prob:.0%} {mode.name}"
        elif prediction.risk_level == 'medium':
            return f"Epoch {prediction.epoch}: Moderate risk - monitoring"
        else:
            return f"Epoch {prediction.epoch}: Training healthy"

    def to_json_summary(self, prediction: FailurePrediction) -> Dict[str, Any]:
        return {
            "epoch": prediction.epoch,
            "timestamp": datetime.now().isoformat(),
            "risk_level": prediction.risk_level,
            "overall_risk": prediction.overall_risk,
            "highest_risk_mode": prediction.get_highest_risk_mode()[0].name,
            "highest_risk_prob": prediction.get_highest_risk_mode()[1],
            "recommendation": prediction.recommendation.action,
            "recommendation_confidence": prediction.recommendation.confidence,
        }