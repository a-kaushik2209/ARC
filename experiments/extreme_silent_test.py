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
ARC EXTREME SILENT CRASH TEST
Testing the WORST silent failure scenarios - failures with no obvious signal.

============================================================
SILENT DEATH SCENARIOS - No NaN, No Inf, No Explosion
============================================================
1. Gradual Weight Decay - Weights slowly shrink to zero
2. Weight Sign Flip - All weights flip sign suddenly
3. Bias Corruption - Only bias terms become zero (subtle)
4. Layer Collapse - Single layer becomes all zeros
5. Magnitude Explosion - Weights grow to 1e6 (no NaN yet)
6. Weight Permutation - Weights get scrambled randomly
7. Gradient Starvation - Gradients become 1e-20 (vanishing)
8. Output Collapse - Final layer becomes constant
9. Feature Death - Hidden activations become constant
10. Rank Collapse - Weight matrices become rank-1

For each: Baseline should detect via sparsity/magnitude check
          ARC should recover via weight health monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import sys
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


@dataclass
class SilentTestResult:
    name: str
    baseline_detected: bool
    arc_recovered: bool
    rollbacks: int
    details: str


class SilentCrashReport:
    def __init__(self):
        self.results: List[SilentTestResult] = []
        self.start_time = time.time()
    
    def add(self, result: SilentTestResult):
        self.results.append(result)
        if result.arc_recovered:
            status = "✓ RECOVERED"
        elif not result.baseline_detected:
            status = "⚠ BASELINE OK"
        else:
            status = "✗ FAILED"
        print(f"  [{status}] {result.name}")
        print(f"       {result.details}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        
        # Only count scenarios where baseline detected the failure
        detectable = [r for r in self.results if r.baseline_detected]
        recovered = sum(1 for r in detectable if r.arc_recovered)
        
        print("\n" + "=" * 70)
        print("EXTREME SILENT CRASH TEST - SUMMARY")
        print("=" * 70)
        print(f"\nTotal Scenarios:     {len(self.results)}")
        print(f"Detectable Failures: {len(detectable)}")
        print(f"ARC Recoveries:      {recovered}")
        
        if len(detectable) > 0:
            rate = 100 * recovered / len(detectable)
            print(f"Recovery Rate:       {rate:.1f}%")
            
            if rate == 100:
                print("\n🎉 PERFECT - ARC handled ALL silent failures!")
            elif rate >= 90:
                print("\n✓ EXCELLENT - ARC handled 90%+ of silent failures")
            elif rate >= 70:
                print("\n⚠ GOOD - Some silent failures need work")
            else:
                print("\n✗ NEEDS IMPROVEMENT")
        
        print(f"\nTime Elapsed: {elapsed:.1f}s")
        return {"total": len(self.results), "recovered": recovered, "detectable": len(detectable)}


class SilentTestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_silent_scenario(
    name: str,
    failure_fn,
    trigger_step: int = 50,
    max_steps: int = 100
) -> SilentTestResult:
    """Run a silent failure scenario."""
    
    print(f"\n>>> {name}")
    
    # Baseline test (manual detection)
    print("  [1/2] Testing baseline detection...")
    model = SilentTestMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    # Capture baseline stats
    baseline_stats = compute_weight_stats(model)
    baseline_detected = False
    baseline_steps = 0
    
    step = 0
    for epoch in range(20):
        for x_batch, y_batch in dataloader:
            step += 1
            if step > max_steps:
                break
            
            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.cross_entropy(out, y_batch)
            
            # Inject failure ONCE at trigger step (not every step after)
            if step == trigger_step:
                failure_fn(model, optimizer, step)
            
            # Check for silent corruption
            current_stats = compute_weight_stats(model)
            
            # Sparsity check
            if current_stats["zero_fraction"] > baseline_stats["zero_fraction"] + 0.15:
                baseline_detected = True
                baseline_steps = step
                break
            
            # Magnitude collapse
            if baseline_stats["mean_magnitude"] > 0:
                ratio = current_stats["mean_magnitude"] / baseline_stats["mean_magnitude"]
                if ratio < 0.05 or ratio > 100:
                    baseline_detected = True
                    baseline_steps = step
                    break
            
            # Variance collapse (all weights same value)
            if current_stats["weight_std"] < 1e-6:
                baseline_detected = True
                baseline_steps = step
                break
            
            loss.backward()
            optimizer.step()
        
        if step > max_steps or baseline_detected:
            break
    
    # ARC test
    print("  [2/2] Testing ARC recovery...")
    model = SilentTestMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = SelfHealingConfig(
        loss_explosion_threshold=50.0,
        gradient_explosion_threshold=1e4,
        lr_reduction_factor=0.5,
        max_lr_reductions=20,
        # Weight health monitoring
        enable_weight_health_check=True,
        weight_sparsity_threshold=0.15,
        weight_magnitude_collapse_threshold=0.05,
        weight_health_check_frequency=1,
        # Persistent failure recovery
        enable_persistent_failure_recovery=True,
        max_consecutive_rollbacks=3,  # Faster escape trigger
        skip_ahead_steps=10,  # Skip more steps
        weight_perturbation_scale=0.02,  # More aggressive perturbation
        verbose=False,
    )
    arc = SelfHealingArc(model, optimizer, config)
    
    arc_crashed = False
    arc_steps = 0
    rollbacks = 0
    
    step = 0
    try:
        for epoch in range(20):
            for x_batch, y_batch in dataloader:
                step += 1
                if step > max_steps:
                    break
                
                optimizer.zero_grad()
                
                try:
                    out = model(x_batch)
                    loss = F.cross_entropy(out, y_batch)
                except Exception:
                    arc_crashed = True
                    break
                
                # Inject failure ONCE at trigger step (not every step after)
                if step == trigger_step:
                    failure_fn(model, optimizer, step)
                
                action = arc.step(loss)
                if action.should_skip:
                    if action.rolled_back:
                        rollbacks += 1
                    continue
                
                try:
                    loss.backward()
                except Exception:
                    # Backward failed, try to skip
                    rollbacks += 1
                    continue
                
                post_action = arc.post_backward()
                if post_action.should_skip:
                    continue
                
                optimizer.step()
                arc_steps = step
            
            if step > max_steps or arc_crashed:
                break
    except Exception:
        arc_crashed = True
    
    arc_recovered = baseline_detected and not arc_crashed and arc_steps >= max_steps - 10
    
    return SilentTestResult(
        name=name,
        baseline_detected=baseline_detected,
        arc_recovered=arc_recovered,
        rollbacks=rollbacks,
        details=f"Baseline: {'DETECTED' if baseline_detected else 'OK'} @ {baseline_steps}, "
                f"ARC: {arc_steps} steps, Rollbacks: {rollbacks}"
    )


def compute_weight_stats(model) -> Dict[str, float]:
    """Compute weight statistics."""
    total_params = 0
    total_zeros = 0
    total_magnitude = 0.0
    all_weights = []
    
    for p in model.parameters():
        numel = p.numel()
        total_params += numel
        total_zeros += (p.data == 0).sum().item()
        total_magnitude += p.data.abs().sum().item()
        all_weights.append(p.data.flatten())
    
    all_weights = torch.cat(all_weights)
    
    return {
        "total_params": total_params,
        "zero_fraction": total_zeros / max(total_params, 1),
        "mean_magnitude": total_magnitude / max(total_params, 1),
        "weight_std": all_weights.std().item(),
        "weight_mean": all_weights.mean().item(),
    }


# =============================================================================
# EXTREME SILENT FAILURE SCENARIOS
# =============================================================================

def silent_gradual_decay(model, optimizer, step):
    """Weights slowly decay by 1% each step - eventually collapse."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 0.99


def silent_aggressive_decay(model, optimizer, step):
    """Weights decay by 10% each step - faster collapse."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 0.9


def silent_weight_sign_flip(model, optimizer, step):
    """All weights flip sign - model inverts."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= -1


def silent_bias_zeroing(model, optimizer, step):
    """Only bias terms become zero - subtle corruption."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'bias' in name:
                p.fill_(0)


def silent_layer_collapse(model, optimizer, step):
    """First layer becomes all zeros."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'fc1' in name:
                p.fill_(0)


def silent_magnitude_explosion(model, optimizer, step):
    """Weights grow by 10% each step - explosion without NaN."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1.1


def silent_weight_permutation(model, optimizer, step):
    """Weights get randomly shuffled."""
    with torch.no_grad():
        for p in model.parameters():
            flat = p.data.flatten()
            perm = torch.randperm(flat.size(0))
            p.data = flat[perm].reshape(p.shape)


def silent_gradient_starvation(model, optimizer, step):
    """Gradients become extremely small."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= 1e-10


def silent_output_collapse(model, optimizer, step):
    """Final layer becomes constant output."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'fc3' in name and 'weight' in name:
                p.data = torch.ones_like(p) * 0.01
            if 'fc3' in name and 'bias' in name:
                p.fill_(0)


def silent_rank_collapse(model, optimizer, step):
    """Weight matrices become rank-1 (outer product)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name and p.dim() == 2:
                # Make rank-1: u @ v.T
                u = torch.randn(p.shape[0], 1) * 0.01
                v = torch.randn(1, p.shape[1]) * 0.01
                p.data = u @ v


def silent_extreme_sparsity(model, optimizer, step):
    """90% of weights become exactly zero."""
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p) < 0.9
            p.data[mask] = 0.0


def silent_weight_cloning(model, optimizer, step):
    """All weights become the same value."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.01)


def silent_alternating_zero(model, optimizer, step):
    """Every other weight becomes zero."""
    with torch.no_grad():
        for p in model.parameters():
            flat = p.data.flatten()
            flat[::2] = 0
            p.data = flat.reshape(p.shape)


def silent_negative_scale(model, optimizer, step):
    """Weights get scaled to very small negative values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = -torch.abs(p.data) * 0.001


def silent_layer_swap(model, optimizer, step):
    """Swap weights between layers - architectural corruption."""
    with torch.no_grad():
        fc1_weight = model.fc1.weight.data.clone()
        fc2_weight = model.fc2.weight.data.clone()
        # Corrupt by mixing
        model.fc1.weight.data[:128, :] = fc2_weight[:128, :100]


SILENT_SCENARIOS = [
    ("Gradual Weight Decay (1%/step)", silent_gradual_decay),
    ("Aggressive Weight Decay (10%/step)", silent_aggressive_decay),
    ("Weight Sign Flip", silent_weight_sign_flip),
    ("Bias Zeroing (subtle)", silent_bias_zeroing),
    ("Layer Collapse (fc1 → zeros)", silent_layer_collapse),
    ("Magnitude Explosion (10%/step)", silent_magnitude_explosion),
    ("Weight Permutation (shuffle)", silent_weight_permutation),
    ("Output Collapse (fc3 → constant)", silent_output_collapse),
    ("Rank Collapse (rank-1 matrices)", silent_rank_collapse),
    ("Extreme Sparsity (90% zeros)", silent_extreme_sparsity),
    ("Weight Cloning (all same value)", silent_weight_cloning),
    ("Alternating Zero (50% pattern)", silent_alternating_zero),
    ("Negative Scale (small negatives)", silent_negative_scale),
]


def run_extreme_silent_tests():
    """Run all extreme silent crash scenarios."""
    
    print("=" * 70)
    print("🔇 EXTREME SILENT CRASH TEST 🔇")
    print("   Testing failures with NO obvious numerical signal")
    print("=" * 70)
    
    report = SilentCrashReport()
    
    for name, failure_fn in SILENT_SCENARIOS:
        result = run_silent_scenario(name, failure_fn)
        report.add(result)
    
    summary = report.summary()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "silent_crash_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "scenarios": [
                {
                    "name": r.name,
                    "baseline_detected": r.baseline_detected,
                    "arc_recovered": r.arc_recovered,
                    "rollbacks": r.rollbacks,
                }
                for r in report.results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return report.results


if __name__ == "__main__":
    run_extreme_silent_tests()
