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

"""
Silent Failure Validation Script v2 - Improved Sensitivity

Improved from v1:
- More sensitive accuracy collapse detection
- Better test case design (80% label noise)
- Proper metric tracking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from arc.signals.silent_detector import (
    SilentCrashDetector, SilentDetectorConfig, SilentFailureType
)


@dataclass
class SilentFailureMetrics:
    """Precision/Recall/F1 for a single failure type."""
    failure_type: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    detection_latency_steps: float


class SimpleClassifier(nn.Module):
    """Simple classifier for testing accuracy collapse."""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def generate_dataset_with_noise(n_samples=1000, input_dim=784, n_classes=10, noise_ratio=0.0):
    """Generate dataset with label noise."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    
    # Add label noise
    n_noisy = int(n_samples * noise_ratio)
    noise_idx = torch.randperm(n_samples)[:n_noisy]
    y[noise_idx] = torch.randint(0, n_classes, (n_noisy,))
    
    return X, y


def compute_accuracy(model, X, y, device):
    """Compute accuracy on dataset."""
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=1)
        acc = (preds == y.to(device)).float().mean().item()
    model.train()
    return acc


def test_accuracy_collapse_with_noise(
    n_runs: int = 10,
    noise_ratio: float = 0.8,  # 80% label noise to cause collapse
    total_steps: int = 150,
) -> SilentFailureMetrics:
    """
    Test accuracy collapse detection on noisy labels.
    
    Model should collapse to ~10% accuracy (random guessing).
    Detector should trigger when accuracy drops significantly.
    """
    print("\n" + "=" * 60)
    print(f"TEST: Accuracy Collapse (Noisy Labels - {int(noise_ratio*100)}%)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tp, fp, tn, fn = 0, 0, 0, 0
    detection_latencies = []
    
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        
        model = SimpleClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # High LR to accelerate collapse
        
        config = SilentDetectorConfig(
            detect_accuracy_collapse=True,
            accuracy_drop_sigma=2.0,  # More sensitive
            min_samples=5,  # Faster detection
            history_size=50,
            verbose=False,
        )
        detector = SilentCrashDetector(config)
        
        # Clean validation set
        X_val, y_val = generate_dataset_with_noise(200, noise_ratio=0.0)
        # Noisy training set
        X_train, y_train = generate_dataset_with_noise(1000, noise_ratio=noise_ratio)
        
        collapse_detected = False
        detection_step = None
        initial_acc = None
        final_acc = None
        
        for step in range(total_steps):
            idx = torch.randint(0, len(X_train), (64,))
            x_batch = X_train[idx].to(device)
            y_batch = y_train[idx].to(device)
            
            optimizer.zero_grad()
            out = model(x_batch)
            loss = nn.functional.cross_entropy(out, y_batch)
            loss.backward()
            optimizer.step()
            
            # Check accuracy every 5 steps
            if step % 5 == 0:
                val_acc = compute_accuracy(model, X_val, y_val, device)
                
                if step == 0:
                    initial_acc = val_acc
                final_acc = val_acc
                
                result = detector.check(loss=loss.item(), val_metric=val_acc)
                
                if result.detected and result.failure_type == SilentFailureType.ACCURACY_COLLAPSE:
                    if not collapse_detected:
                        collapse_detected = True
                        detection_step = step
        
        # Determine if actual collapse occurred
        actual_collapse = final_acc < 0.15  # Below 15% = collapsed to random
        
        if actual_collapse and collapse_detected:
            tp += 1
            if detection_step:
                detection_latencies.append(detection_step)
        elif actual_collapse and not collapse_detected:
            fn += 1
        elif not actual_collapse and collapse_detected:
            fp += 1
        else:
            tn += 1
        
        status = "TP" if (actual_collapse and collapse_detected) else \
                 "FN" if (actual_collapse and not collapse_detected) else \
                 "FP" if (not actual_collapse and collapse_detected) else "TN"
        
        print(f"  Run {run+1}/{n_runs}: Final acc {final_acc:.1%}, Collapsed: {actual_collapse}, Detected: {collapse_detected} [{status}]")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = SilentFailureMetrics(
        failure_type="accuracy_collapse",
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        detection_latency_steps=np.mean(detection_latencies) if detection_latencies else 0,
    )
    
    print(f"\n  Results:")
    print(f"    True Positives: {tp}")
    print(f"    False Negatives: {fn}")
    print(f"    Precision: {precision:.2%}")
    print(f"    Recall: {recall:.2%}")
    print(f"    F1: {f1:.2%}")
    
    return metrics


def test_false_positive_on_stable(
    n_runs: int = 10,
    total_steps: int = 100,
) -> float:
    """Test false positive rate on stable (clean) training runs."""
    print("\n" + "=" * 60)
    print("TEST: False Positive Rate (Stable Runs)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_runs_with_detection = 0
    
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        
        model = SimpleClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        config = SilentDetectorConfig(
            detect_accuracy_collapse=True,
            accuracy_drop_sigma=2.0,
            min_samples=5,
            verbose=False,
        )
        detector = SilentCrashDetector(config)
        
        # Clean dataset
        X_train, y_train = generate_dataset_with_noise(1000, noise_ratio=0.0)
        X_val, y_val = generate_dataset_with_noise(200, noise_ratio=0.0)
        
        detected_anything = False
        
        for step in range(total_steps):
            idx = torch.randint(0, len(X_train), (64,))
            x_batch = X_train[idx].to(device)
            y_batch = y_train[idx].to(device)
            
            optimizer.zero_grad()
            out = model(x_batch)
            loss = nn.functional.cross_entropy(out, y_batch)
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                val_acc = compute_accuracy(model, X_val, y_val, device)
                result = detector.check(loss=loss.item(), val_metric=val_acc)
                
                if result.detected and result.failure_type == SilentFailureType.ACCURACY_COLLAPSE:
                    detected_anything = True
        
        if detected_anything:
            total_runs_with_detection += 1
        
        print(f"  Run {run+1}/{n_runs}: False alarm: {detected_anything}")
    
    fpr = total_runs_with_detection / n_runs
    print(f"\n  False Positive Rate: {fpr:.1%} ({total_runs_with_detection}/{n_runs} runs)")
    
    return fpr


def run_validation():
    """Run complete silent failure validation."""
    print("=" * 70)
    print("SILENT FAILURE VALIDATION v2")
    print("=" * 70)
    
    # Test 1: Accuracy collapse with noisy labels
    collapse_metrics = test_accuracy_collapse_with_noise(n_runs=10)
    
    # Test 2: False positive rate
    fpr = test_false_positive_on_stable(n_runs=10)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Accuracy Collapse Detection:")
    print(f"  Precision: {collapse_metrics.precision:.1%}")
    print(f"  Recall: {collapse_metrics.recall:.1%}")
    print(f"  F1: {collapse_metrics.f1:.1%}")
    print(f"\nFalse Positive Rate (stable runs): {fpr:.1%}")
    
    results = {
        "accuracy_collapse": asdict(collapse_metrics),
        "false_positive_rate": fpr,
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Save
    with open('silent_failure_metrics_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: silent_failure_metrics_v2.json")
    
    return results


if __name__ == '__main__':
    results = run_validation()