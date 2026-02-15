#!/usr/bin/env python

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
V3 Benchmark: Aggressive Detection Mode

Fixed issues:
1. Threshold logic (higher = more sensitive) 
2. More aggressive early detection
3. Better signal weights

Run: python experiments/v3_benchmark.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_data(n_samples=1000, n_features=20, n_classes=5, seed=42):
    set_seed(seed)
    X = torch.randn(n_samples, n_features)
    centers = torch.randn(n_classes, n_features) * 2
    y = torch.randint(0, n_classes, (n_samples,))
    for i in range(n_classes):
        mask = y == i
        X[mask] = X[mask] + centers[i]
    return X, y


def create_model(n_features=20, n_classes=5):
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )


class AggressiveDetector:
    """
    Aggressive failure detector - prioritizes recall.
    
    Philosophy: Better to warn early than miss failures.
    """
    
    def __init__(self):
        self.loss_history = []
        self.grad_history = []
        self.step = 0
        self.detected = False
        self.detection_reason = None
    
    def update(self, loss, grad_norm):
        """Update and check for failures."""
        # Handle bad values
        if math.isnan(loss) or math.isinf(loss):
            self.detected = True
            self.detection_reason = "NaN/Inf loss"
            return True
        
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            self.detected = True
            self.detection_reason = "NaN/Inf gradient"
            return True
        
        self.loss_history.append(loss)
        self.grad_history.append(grad_norm)
        self.step += 1
        
        # Check multiple failure conditions
        if self._check_gradient_explosion(grad_norm):
            self.detected = True
            self.detection_reason = "Gradient explosion"
            return True
        
        if self._check_gradient_vanishing(grad_norm):
            self.detected = True
            self.detection_reason = "Gradient vanishing"
            return True
        
        if self._check_loss_explosion():
            self.detected = True
            self.detection_reason = "Loss explosion"
            return True
        
        if self._check_loss_increase():
            self.detected = True
            self.detection_reason = "Loss increasing"
            return True
        
        if self._check_instability():
            self.detected = True
            self.detection_reason = "Training instability"
            return True
        
        return False
    
    def _check_gradient_explosion(self, grad_norm):
        """Detect exploding gradients."""
        if grad_norm > 50:  # Aggressive threshold
            return True
        
        if len(self.grad_history) >= 3:
            recent = self.grad_history[-3:]
            if recent[-1] > recent[0] * 5:  # 5x increase
                return True
        
        return False
    
    def _check_gradient_vanishing(self, grad_norm):
        """Detect vanishing gradients."""
        if grad_norm < 1e-5:
            return True
        
        # Gradients decreasing rapidly
        if len(self.grad_history) >= 5:
            recent = self.grad_history[-5:]
            if recent[-1] < recent[0] * 0.01:  # 100x decrease
                return True
        
        return False
    
    def _check_loss_explosion(self):
        """Detect loss explosion."""
        if not self.loss_history:
            return False
        
        if self.loss_history[-1] > 100:  # Very high loss
            return True
        
        if len(self.loss_history) >= 3:
            if self.loss_history[-1] > self.loss_history[-2] * 3:
                return True
        
        return False
    
    def _check_loss_increase(self):
        """Detect sustained loss increase."""
        if len(self.loss_history) < 5:
            return False
        
        recent = self.loss_history[-5:]
        
        # Check if most recent losses are increasing
        increases = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        
        if increases >= 4:  # 4 out of 4 increases
            return True
        
        # Check if final loss >> initial loss
        if recent[-1] > recent[0] * 2:
            return True
        
        return False
    
    def _check_instability(self):
        """Detect oscillating/unstable training."""
        if len(self.loss_history) < 10:
            return False
        
        recent = self.loss_history[-10:]
        
        # Check variance - high variance = unstable
        mean_loss = np.mean(recent)
        std_loss = np.std(recent)
        
        cv = std_loss / (mean_loss + 1e-10)
        
        if cv > 1.0:  # Coefficient of variation > 1
            return True
        
        return False
    
    def is_failing(self):
        return self.detected
    
    def reset(self):
        self.loss_history = []
        self.grad_history = []
        self.step = 0
        self.detected = False
        self.detection_reason = None


class SimpleBaseline:
    """Simple rule-based baseline."""
    
    def __init__(self):
        self.losses = []
        self.grads = []
    
    def update(self, loss, grad_norm):
        self.losses.append(loss)
        self.grads.append(grad_norm)
        
        if grad_norm > 1000 or grad_norm < 1e-8:
            return True
        if len(self.losses) > 3 and self.losses[-1] > self.losses[-2] * 2:
            return True
        return False
    
    def reset(self):
        self.losses = []
        self.grads = []


def run_experiment(failure_mode="none", seed=42, n_epochs=25):
    """Run single experiment."""
    set_seed(seed)
    
    X, y = create_data(seed=seed)
    split = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=32, shuffle=True)
    
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    detector = AggressiveDetector()
    baseline = SimpleBaseline()
    
    result = {
        "failure_mode": failure_mode,
        "seed": seed,
        "omega_detected": False,
        "omega_epoch": None,
        "omega_reason": None,
        "baseline_detected": False,
        "baseline_epoch": None,
        "actual_failure_epoch": None,
    }
    
    for epoch in range(n_epochs):
        # Induce failure at epoch 10
        if epoch == 10 and failure_mode != "none":
            result["actual_failure_epoch"] = epoch
            
            if failure_mode == "vanishing":
                for p in model.parameters():
                    p.data *= 0.001
            elif failure_mode == "lr_explosion":
                for g in optimizer.param_groups:
                    g['lr'] = 1.0
            elif failure_mode == "gradient_explosion":
                pass  # Applied in loop
            elif failure_mode == "mode_collapse":
                with torch.no_grad():
                    for p in list(model.parameters())[-2:]:
                        p.data.fill_(0.1)
            elif failure_mode == "grokking_risk":
                for g in optimizer.param_groups:
                    g['weight_decay'] = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                result["actual_failure_epoch"] = epoch
                # Detector should catch this
                detector.update(float('inf'), 0)
                if detector.is_failing() and not result["omega_detected"]:
                    result["omega_detected"] = True
                    result["omega_epoch"] = epoch
                    result["omega_reason"] = detector.detection_reason
                return result
            
            loss.backward()
            
            # Apply gradient explosion
            if failure_mode == "gradient_explosion" and epoch >= 10:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad *= 50
            
            grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            
            # Update detectors
            detector.update(loss.item(), grad_norm)
            baseline_detect = baseline.update(loss.item(), grad_norm)
            
            # Record first detection
            if detector.is_failing() and not result["omega_detected"]:
                result["omega_detected"] = True
                result["omega_epoch"] = epoch
                result["omega_reason"] = detector.detection_reason
            
            if baseline_detect and not result["baseline_detected"]:
                result["baseline_detected"] = True
                result["baseline_epoch"] = epoch
            
            optimizer.step()
    
    return result


def run_benchmark():
    """Run comprehensive benchmark."""
    print("=" * 70)
    print("V3 BENCHMARK: Aggressive ARC Detection")
    print("=" * 70)
    
    failure_modes = [
        "none",
        "vanishing",
        "lr_explosion",
        "gradient_explosion",
        "mode_collapse",
        "grokking_risk",
    ]
    
    n_trials = 10
    all_results = defaultdict(list)
    
    for mode in failure_modes:
        print(f"\n{mode.upper()}")
        print("-" * 40)
        
        for trial in range(n_trials):
            result = run_experiment(failure_mode=mode, seed=42+trial)
            all_results[mode].append(result)
            
            should_detect = mode != "none"
            omega_ok = result["omega_detected"] == should_detect
            baseline_ok = result["baseline_detected"] == should_detect
            
            o_mark = "✓" if omega_ok else "✗"
            b_mark = "✓" if baseline_ok else "✗"
            reason = result.get("omega_reason", "")[:25] if result["omega_detected"] else ""
            
            print(f"  {trial+1:2d}: Ω={o_mark} B={b_mark} | {reason}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    healthy = all_results["none"]
    failures = [r for mode in failure_modes[1:] for r in all_results[mode]]
    
    o_tp = sum(1 for r in failures if r["omega_detected"])
    o_fn = sum(1 for r in failures if not r["omega_detected"])
    o_tn = sum(1 for r in healthy if not r["omega_detected"])
    o_fp = sum(1 for r in healthy if r["omega_detected"])
    
    b_tp = sum(1 for r in failures if r["baseline_detected"])
    b_fn = sum(1 for r in failures if not r["baseline_detected"])
    b_tn = sum(1 for r in healthy if not r["baseline_detected"])
    b_fp = sum(1 for r in healthy if r["baseline_detected"])
    
    def metrics(tp, fp, tn, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        return acc, prec, rec, f1
    
    o_acc, o_prec, o_rec, o_f1 = metrics(o_tp, o_fp, o_tn, o_fn)
    b_acc, b_prec, b_rec, b_f1 = metrics(b_tp, b_fp, b_tn, b_fn)
    
    print(f"\n{'Method':<20} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 60)
    print(f"{'ARC v3':<20} {o_acc*100:>5.1f}%  {o_prec*100:>5.1f}%  {o_rec*100:>5.1f}%   {o_f1*100:>5.1f}%")
    print(f"{'Baseline':<20} {b_acc*100:>5.1f}%  {b_prec*100:>5.1f}%  {b_rec*100:>5.1f}%   {b_f1*100:>5.1f}%")
    
    print(f"\nARC Confusion Matrix:")
    print(f"   TP={o_tp} FP={o_fp} TN={o_tn} FN={o_fn}")
    
    # Winner
    print("\n" + "=" * 70)
    if o_f1 > b_f1:
        print(f"Ω-NET WINS! +{(o_f1-b_f1)*100:.1f} F1 points")
    elif o_f1 == b_f1:
        print("TIE")
    else:
        print(f"Baseline ahead by {(b_f1-o_f1)*100:.1f} F1 points")
    print("=" * 70)
    
    # Save
    summary = {
        "arc_network": {"accuracy": o_acc, "precision": o_prec, "recall": o_rec, "f1": o_f1,
                      "tp": o_tp, "fp": o_fp, "tn": o_tn, "fn": o_fn},
        "baseline": {"accuracy": b_acc, "precision": b_prec, "recall": b_rec, "f1": b_f1},
        "winner": "arc_network" if o_f1 > b_f1 else "baseline" if b_f1 > o_f1 else "tie",
        "total_experiments": 60,
    }
    
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/v3_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to experiments/v3_results.json")
    
    return summary


if __name__ == "__main__":
    run_benchmark()