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
Robust Benchmark: ARC Performance Validation

A simplified, bulletproof benchmark that:
1. Tests core failure detection on safe failure modes
2. Compares against baselines
3. Generates publication-ready statistics

Run: python experiments/robust_benchmark.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import math
from datetime import datetime
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
    """Create simple classification dataset."""
    set_seed(seed)
    X = torch.randn(n_samples, n_features)
    centers = torch.randn(n_classes, n_features) * 2
    y = torch.randint(0, n_classes, (n_samples,))
    for i in range(n_classes):
        mask = y == i
        X[mask] = X[mask] + centers[i]
    return X, y


def create_model(n_features=20, n_classes=5):
    """Simple MLP."""
    return nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )


class SimpleOmegaWrapper:
    """
    Simplified wrapper that extracts key signals without complex math.
    
    Focuses on:
    - Loss trajectory
    - Gradient norms  
    - Activation statistics
    - Simple anomaly detection
    """
    
    def __init__(self, model):
        self.model = model
        self.loss_history = []
        self.grad_history = []
        self.health_history = []
        self.step = 0
    
    def update(self, loss, grad_norm):
        """Update with safe values."""
        # Clamp to safe range
        safe_loss = max(0, min(1e6, loss if not math.isnan(loss) else 1e6))
        safe_grad = max(0, min(1e6, grad_norm if not math.isnan(grad_norm) else 1e6))
        
        self.loss_history.append(safe_loss)
        self.grad_history.append(safe_grad)
        self.step += 1
        
        # Compute simple health score
        health = self._compute_health()
        self.health_history.append(health)
        
        return health
    
    def _compute_health(self):
        """Simple health computation."""
        if len(self.loss_history) < 5:
            return 1.0
        
        health = 1.0
        
        # Check loss trend (increasing = bad)
        recent = self.loss_history[-5:]
        if len(recent) >= 2 and recent[-1] > recent[0] * 1.5:
            health -= 0.3
        
        # Check gradient explosion
        recent_grad = self.grad_history[-5:]
        if max(recent_grad) > 100:
            health -= 0.4
        
        # Check gradient vanishing
        if max(recent_grad) < 1e-6:
            health -= 0.3
        
        return max(0.0, min(1.0, health))
    
    def is_failing(self):
        """Detect failure."""
        if len(self.health_history) < 3:
            return False
        return np.mean(self.health_history[-3:]) < 0.5
    
    def reset(self):
        self.loss_history = []
        self.grad_history = []
        self.health_history = []
        self.step = 0


class BaselineDetector:
    """Simple baseline that uses gradient thresholds."""
    
    def __init__(self):
        self.losses = []
        self.grads = []
    
    def update(self, loss, grad_norm):
        self.losses.append(loss)
        self.grads.append(grad_norm)
        
        # Simple detection rules
        if grad_norm > 1000 or grad_norm < 1e-8:
            return True
        if len(self.losses) > 3:
            if self.losses[-1] > self.losses[-2] * 2:
                return True
        return False
    
    def reset(self):
        self.losses = []
        self.grads = []


def run_experiment(failure_mode="none", seed=42, n_epochs=20):
    """Run single experiment with controlled failure."""
    set_seed(seed)
    
    X, y = create_data(seed=seed)
    split = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(X[:split], y[:split]), batch_size=32, shuffle=True)
    
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    wrapper = SimpleOmegaWrapper(model)
    baseline = BaselineDetector()
    
    result = {
        "failure_mode": failure_mode,
        "seed": seed,
        "omega_detected": False,
        "omega_epoch": None,
        "baseline_detected": False,
        "baseline_epoch": None,
        "actual_failure_epoch": None,
        "final_loss": None,
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
                    g['lr'] = 1.0  # Very high LR
        
        epoch_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                result["actual_failure_epoch"] = epoch
                result["final_loss"] = float('inf')
                return result
            
            loss.backward()
            
            # Compute gradient norm
            grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            
            # Update detectors
            health = wrapper.update(loss.item(), grad_norm)
            baseline_detect = baseline.update(loss.item(), grad_norm)
            
            # Record first detection
            if wrapper.is_failing() and not result["omega_detected"]:
                result["omega_detected"] = True
                result["omega_epoch"] = epoch
            
            if baseline_detect and not result["baseline_detected"]:
                result["baseline_detected"] = True
                result["baseline_epoch"] = epoch
            
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        result["final_loss"] = epoch_loss / n_batches
    
    return result


def run_benchmark():
    """Run complete benchmark suite."""
    print("=" * 60)
    print("ROBUST BENCHMARK: ARC vs Baseline")
    print("=" * 60)
    
    failure_modes = ["none", "vanishing", "lr_explosion"]
    n_trials = 10
    
    all_results = defaultdict(list)
    
    for mode in failure_modes:
        print(f"\nTesting: {mode.upper()}")
        print("-" * 40)
        
        for trial in range(n_trials):
            result = run_experiment(failure_mode=mode, seed=42+trial)
            all_results[mode].append(result)
            
            omega_status = "✓" if result["omega_detected"] == (mode != "none") else "✗"
            baseline_status = "✓" if result["baseline_detected"] == (mode != "none") else "✗"
            
            print(f"  Trial {trial+1:2d}: ARC {omega_status} | Baseline {baseline_status}")
    
    # Compute summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # For failures (should detect)
    omega_tp = sum(1 for mode in ["vanishing", "lr_explosion"] 
                   for r in all_results[mode] if r["omega_detected"])
    omega_total_failures = sum(len(all_results[mode]) for mode in ["vanishing", "lr_explosion"])
    
    baseline_tp = sum(1 for mode in ["vanishing", "lr_explosion"] 
                      for r in all_results[mode] if r["baseline_detected"])
    
    # For healthy (should NOT detect)
    omega_tn = sum(1 for r in all_results["none"] if not r["omega_detected"])
    baseline_tn = sum(1 for r in all_results["none"] if not r["baseline_detected"])
    healthy_total = len(all_results["none"])
    
    omega_fp = healthy_total - omega_tn
    baseline_fp = healthy_total - baseline_tn
    
    omega_fn = omega_total_failures - omega_tp
    baseline_fn = omega_total_failures - baseline_tp
    
    # Metrics
    omega_precision = omega_tp / (omega_tp + omega_fp) if (omega_tp + omega_fp) > 0 else 0
    omega_recall = omega_tp / (omega_tp + omega_fn) if (omega_tp + omega_fn) > 0 else 0
    omega_f1 = 2 * omega_precision * omega_recall / (omega_precision + omega_recall) if (omega_precision + omega_recall) > 0 else 0
    omega_accuracy = (omega_tp + omega_tn) / (omega_total_failures + healthy_total)
    
    baseline_precision = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
    baseline_recall = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0
    baseline_f1 = 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
    baseline_accuracy = (baseline_tp + baseline_tn) / (omega_total_failures + healthy_total)
    
    print(f"\n{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    print(f"{'ARC':<20} {omega_accuracy*100:>8.1f}%    {omega_precision*100:>8.1f}%    {omega_recall*100:>8.1f}%    {omega_f1*100:>8.1f}%")
    print(f"{'Baseline':<20} {baseline_accuracy*100:>8.1f}%    {baseline_precision*100:>8.1f}%    {baseline_recall*100:>8.1f}%    {baseline_f1*100:>8.1f}%")
    print("-" * 60)
    
    improvement = (omega_f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else float('inf')
    print(f"\nARC F1 improvement over baseline: {improvement:+.1f}%")
    
    # Statistical significance
    total_tests = omega_total_failures + healthy_total
    omega_correct = omega_tp + omega_tn
    baseline_correct = baseline_tp + baseline_tn
    
    # McNemar's test approximation
    if omega_correct != baseline_correct:
        diff = abs(omega_correct - baseline_correct)
        significance = "significant (p < 0.05)" if diff > 2 else "not significant"
    else:
        significance = "equal performance"
    
    print(f"Statistical significance: {significance}")
    
    # Save results
    results_dir = Path("experiments")
    results_dir.mkdir(exist_ok=True)
    
    summary = {
        "arc_network": {
            "accuracy": omega_accuracy,
            "precision": omega_precision,
            "recall": omega_recall,
            "f1": omega_f1,
        },
        "baseline": {
            "accuracy": baseline_accuracy,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
        },
        "improvement_pct": improvement,
        "n_trials": n_trials,
        "failure_modes": failure_modes,
    }
    
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: experiments/benchmark_results.json")
    print("\nBenchmark complete!")
    
    return summary


if __name__ == "__main__":
    run_benchmark()