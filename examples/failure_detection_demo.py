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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.learning.simulator import FailureSimulator
from arc.learning.labeler import TrajectoryLabeler
from arc.config import FailureMode
from arc.api.report import ReportGenerator


def demo_failure_detection():
    
    print("=" * 70)
    print("Arc Failure Detection Demo")
    print("=" * 70)
    
    simulator = FailureSimulator(seed=42)
    labeler = TrajectoryLabeler()
    reporter = ReportGenerator()
    
    demos = [
        ("Divergence", simulator.generate_divergence_trajectory(severity=0.8)),
        ("Vanishing Gradients", simulator.generate_vanishing_gradient_trajectory(severity=0.7)),
        ("Exploding Gradients", simulator.generate_exploding_gradient_trajectory(severity=0.75)),
        ("Representation Collapse", simulator.generate_representation_collapse_trajectory(severity=0.8)),
        ("Overfitting", simulator.generate_overfitting_trajectory(severity=0.7)),
        ("Healthy Training", simulator.generate_successful_trajectory()),
    ]
    
    for name, trajectory in demos:
        print(f"\n{'─' * 70}")
        print(f"{name}")
        print(f"{'─' * 70}")
        
        print(f"   Trajectory length: {trajectory.n_epochs} epochs")
        print(f"   Ground truth failure: {trajectory.failure_mode}")
        print(f"   Failure epoch: {trajectory.failure_epoch}")
        print(f"   Severity: {trajectory.severity:.1%}")
        
        print("\n   Detection timeline:")
        detection_points = [0.25, 0.5, 0.75, 1.0]
        
        for fraction in detection_points:
            n_epochs = int(trajectory.n_epochs * fraction)
            partial_signals = trajectory.signals[:n_epochs]
            
            if not partial_signals:
                continue
            
            label = labeler.label_trajectory(partial_signals)
            
            if label.is_failure:
                status = f"{label.mode.name} detected (conf: {label.confidence:.1%})"
            else:
                status = f"No failure detected (risk: {1-label.confidence:.1%})"
            
            print(f"   Epoch {n_epochs:3d} ({fraction:.0%}): {status}")
        
        final_label = labeler.label_trajectory(trajectory.signals)
        
        if trajectory.is_failure:
            if final_label.mode == trajectory.failure_mode:
                detection_epoch = final_label.failure_epoch or trajectory.n_epochs
                lead_time = (trajectory.failure_epoch or trajectory.n_epochs) - detection_epoch
                print(f"\n   ✓ Correctly detected with {max(0, lead_time)} epoch lead time")
            elif final_label.is_failure:
                print(f"\n   ⚠ Misclassified as {final_label.mode}")
            else:
                print(f"\n   ✗ Missed detection")
        else:
            if final_label.is_failure:
                print(f"\n   ✗ False alarm: {final_label.mode}")
            else:
                print(f"\n   ✓ Correctly identified as healthy")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def demo_signal_analysis():
    
    print("\n" + "=" * 70)
    print("Signal Analysis Demo")
    print("=" * 70)
    
    simulator = FailureSimulator(seed=42)
    trajectory = simulator.generate_divergence_trajectory(severity=0.8, n_epochs=30)
    
    print("\nSignal Evolution (Divergence Trajectory):")
    print("-" * 70)
    print(f"{'Epoch':>6} │ {'Loss':>10} │ {'Grad Norm':>10} │ {'LR':>10} │ Status")
    print("-" * 70)
    
    for i, signals in enumerate(trajectory.signals):
        loss = signals.get("loss", {}).get("epoch", {}).get("train_loss", 0)
        grad_norm = signals.get("gradient", {}).get("global", {}).get("total_grad_norm_l2", 0)
        lr = signals.get("optimizer", {}).get("global", {}).get("effective_lr", 0)
        
        if trajectory.failure_epoch and i >= trajectory.failure_epoch:
            status = "Diverging"
        elif trajectory.failure_epoch and i >= trajectory.failure_epoch - 5:
            status = "Warning"
        else:
            status = "OK"
        
        print(f"{i:>6} │ {loss:>10.4f} │ {grad_norm:>10.4f} │ {lr:>10.6f} │ {status}")
    
    print("-" * 70)


if __name__ == '__main__':
    demo_failure_detection()
    demo_signal_analysis()