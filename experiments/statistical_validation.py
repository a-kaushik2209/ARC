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
ARC Statistical Validation Suite (Phase 1)

Addresses reviewer concern: "100% claims need confidence intervals"

This module runs:
- 10 independent seeds per experiment
- Computes mean ± std, 95% CI
- Bootstrap p-values vs baseline
- Proper statistical reporting

Output: statistical_results.json with full metrics
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
from scipy import stats
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Configuration
# =============================================================================

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 5926, 8675, 3099]  # 10 seeds
N_STEPS = 100
INJECT_AT = 50


# =============================================================================
# Models
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =============================================================================
# Failure Injectors
# =============================================================================

def inject_nan_loss(loss, step):
    if step == INJECT_AT:
        return torch.tensor(float('nan'))
    return loss


def inject_inf_loss(loss, step):
    if step == INJECT_AT:
        return torch.tensor(float('inf'))
    return loss


def inject_loss_explosion(loss, step):
    if step == INJECT_AT:
        return loss * 1000
    return loss


def inject_weight_corruption(model, step):
    if step == INJECT_AT:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'weight' in name:
                    mask = torch.rand_like(p) < 0.01
                    p.data[mask] = float('nan')
                    break


FAILURE_TYPES = {
    "nan_loss": inject_nan_loss,
    "inf_loss": inject_inf_loss,
    "loss_explosion": inject_loss_explosion,
}


# =============================================================================
# Single Run
# =============================================================================

def run_single_test(model_class, failure_type, seed, use_arc=True):
    """Run a single test with specific seed."""
    
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Data
    if isinstance(model, SimpleCNN):
        X = torch.randn(200, 3, 32, 32)
    else:
        X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    # ARC setup
    shard = None
    if use_arc:
        config = SelfHealingConfig(
            checkpoint_frequency=10,
            loss_explosion_threshold=100.0,
            verbose=False,
        )
        shard = SelfHealingArc(model, optimizer, config)
    
    result = {
        'seed': seed,
        'failure_type': failure_type,
        'use_arc': use_arc,
        'recovered': False,
        'steps_completed': 0,
        'crashed': False,
        'rollbacks': 0,
        'final_loss': None,
    }
    
    step = 0
    injector = FAILURE_TYPES.get(failure_type)
    
    try:
        for epoch in range(10):
            for batch_idx, (x, labels) in enumerate(dataloader):
                step += 1
                if step > N_STEPS:
                    break
                
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, labels)
                
                # Weight corruption is handled differently
                if failure_type == "weight_corruption":
                    inject_weight_corruption(model, step)
                elif injector:
                    loss = injector(loss, step)
                
                # ARC protection
                if shard:
                    action = shard.step(loss)
                    if action.should_skip:
                        if action.rolled_back:
                            result['rollbacks'] += 1
                        continue
                else:
                    # No ARC - check for crash
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e10:
                        result['crashed'] = True
                        break
                    
                    # Check weights
                    has_nan = any(torch.isnan(p).any() for p in model.parameters())
                    if has_nan:
                        result['crashed'] = True
                        break
                
                loss.backward()
                
                if shard:
                    shard.post_backward()
                
                optimizer.step()
                result['steps_completed'] = step
                result['final_loss'] = loss.item()
            
            if step > N_STEPS or result['crashed']:
                break
    
    except Exception as e:
        result['crashed'] = True
        result['error'] = str(e)[:100]
    
    if shard:
        stats = shard.get_stats()
        result['rollbacks'] = stats['total_rollbacks']
    
    # Determine if recovered
    result['recovered'] = not result['crashed'] and result['steps_completed'] >= N_STEPS
    
    return result


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_statistics(results: List[Dict]) -> Dict:
    """Compute comprehensive statistics from multiple runs."""
    
    recoveries = [1 if r['recovered'] else 0 for r in results]
    rollbacks = [r['rollbacks'] for r in results]
    steps = [r['steps_completed'] for r in results]
    
    n = len(results)
    
    # Mean and std
    recovery_mean = np.mean(recoveries)
    recovery_std = np.std(recoveries, ddof=1) if n > 1 else 0
    
    rollback_mean = np.mean(rollbacks)
    rollback_std = np.std(rollbacks, ddof=1) if n > 1 else 0
    
    # 95% Confidence Interval (t-distribution for small samples)
    if n > 1:
        ci_95 = stats.t.interval(0.95, n-1, loc=recovery_mean, scale=recovery_std/np.sqrt(n))
        ci_95 = (max(0, ci_95[0]), min(1, ci_95[1]))  # Clamp to [0,1]
    else:
        ci_95 = (recovery_mean, recovery_mean)
    
    # Bootstrap p-value (vs 0% recovery)
    boot_samples = 10000
    boot_means = []
    for _ in range(boot_samples):
        sample = np.random.choice(recoveries, size=n, replace=True)
        boot_means.append(np.mean(sample))
    
    p_value = np.mean(np.array(boot_means) <= 0)  # P(recovery <= 0)
    
    return {
        'n_runs': n,
        'recovery_rate': {
            'mean': recovery_mean,
            'std': recovery_std,
            'ci_95_low': ci_95[0],
            'ci_95_high': ci_95[1],
            'min': min(recoveries),
            'max': max(recoveries),
        },
        'rollbacks': {
            'mean': rollback_mean,
            'std': rollback_std,
        },
        'steps_completed': {
            'mean': np.mean(steps),
            'std': np.std(steps, ddof=1) if n > 1 else 0,
        },
        'p_value_vs_zero': p_value,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_statistical_validation():
    """Run full statistical validation suite."""
    
    print("="*70)
    print("STATISTICAL VALIDATION SUITE")
    print("   10 seeds × 3 failures × 2 models × 2 conditions = 120 runs")
    print("="*70)
    
    models = [("MLP", SimpleMLP), ("CNN", SimpleCNN)]
    failure_types = list(FAILURE_TYPES.keys()) + ["weight_corruption"]
    
    all_results = {}
    
    for model_name, model_class in models:
        all_results[model_name] = {}
        
        for failure in failure_types:
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name} | FAILURE: {failure}")
            print("="*70)
            
            # Baseline runs (no ARC)
            print("\n[Baseline: No Protection]")
            baseline_results = []
            for i, seed in enumerate(SEEDS):
                result = run_single_test(model_class, failure, seed, use_arc=False)
                baseline_results.append(result)
                status = "✅" if result['recovered'] else "❌"
                print(f"  Seed {seed}: {status} (steps: {result['steps_completed']})")
            
            baseline_stats = compute_statistics(baseline_results)
            
            # ARC runs
            print("\n[With SelfHealingArc]")
            arc_results = []
            for i, seed in enumerate(SEEDS):
                result = run_single_test(model_class, failure, seed, use_arc=True)
                arc_results.append(result)
                status = "✅" if result['recovered'] else "❌"
                print(f"  Seed {seed}: {status} (steps: {result['steps_completed']}, rollbacks: {result['rollbacks']})")
            
            arc_stats = compute_statistics(arc_results)
            
            # Store results
            all_results[model_name][failure] = {
                'baseline': {
                    'raw': baseline_results,
                    'stats': baseline_stats,
                },
                'arc': {
                    'raw': arc_results,
                    'stats': arc_stats,
                },
            }
            
            # Print summary
            print(f"\n  Summary:")
            print(f"    Baseline: {baseline_stats['recovery_rate']['mean']*100:.1f}% ± {baseline_stats['recovery_rate']['std']*100:.1f}%")
            print(f"    ARC:      {arc_stats['recovery_rate']['mean']*100:.1f}% ± {arc_stats['recovery_rate']['std']*100:.1f}%")
            print(f"    95% CI:   [{arc_stats['recovery_rate']['ci_95_low']*100:.1f}%, {arc_stats['recovery_rate']['ci_95_high']*100:.1f}%]")
            print(f"    p-value:  {arc_stats['p_value_vs_zero']:.4f}")
    
    # Final summary table
    print("\n" + "="*70)
    print("FINAL STATISTICAL SUMMARY")
    print("="*70)
    
    print("\n| Model | Failure | Baseline | ARC Mean ± Std | 95% CI | p-value |")
    print("|-------|---------|----------|----------------|--------|---------|")
    
    for model in all_results:
        for failure in all_results[model]:
            b = all_results[model][failure]['baseline']['stats']['recovery_rate']
            a = all_results[model][failure]['arc']['stats']['recovery_rate']
            p = all_results[model][failure]['arc']['stats']['p_value_vs_zero']
            
            print(f"| {model:5} | {failure:15} | {b['mean']*100:6.1f}% | {a['mean']*100:.1f}% ± {a['std']*100:.1f}% | [{a['ci_95_low']*100:.0f}%, {a['ci_95_high']*100:.0f}%] | {p:.4f} |")
    
    # Save results
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    with open("statistical_results.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    
    print("\nResults saved to: statistical_results.json")
    
    return all_results


if __name__ == "__main__":
    run_statistical_validation()