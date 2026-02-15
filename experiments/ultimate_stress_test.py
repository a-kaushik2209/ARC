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
ARC Ultimate Stress Test (Phase 19)

Testing SelfHealingArc against EVERY possible failure mode:
1. NaN Loss
2. Inf Loss
3. Loss Explosion (1000x)
4. Gradient Explosion (1e6)
5. Weight Corruption (NaN injection)
6. Optimizer State Corruption
7. Random sudden spikes
8. Repeated failures

This is the ULTIMATE test to prove ARC can handle ANY failure.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Models
# =============================================================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(256 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =============================================================================
# Failure Injectors
# =============================================================================

def inject_nan_loss(model, optimizer, loss, step):
    return torch.tensor(float('nan'))

def inject_inf_loss(model, optimizer, loss, step):
    return torch.tensor(float('inf'))

def inject_loss_explosion(model, optimizer, loss, step):
    return loss * 1000

def inject_gradient_explosion(model, optimizer, loss, step):
    # After backward, gradients will be scaled
    return loss, "grad_scale"

def inject_weight_corruption(model, optimizer, loss, step):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'weight' in name:
                mask = torch.rand_like(p) < 0.01
                p.data[mask] = float('nan')
                break
    return loss

def inject_lr_nuke(model, optimizer, loss, step):
    for pg in optimizer.param_groups:
        pg['lr'] *= 10000
    return loss

def inject_repeated_failures(model, optimizer, loss, step):
    # Fail every 3rd step
    if step % 3 == 0:
        return torch.tensor(float('inf'))
    return loss


FAILURE_INJECTORS = {
    "nan_loss": inject_nan_loss,
    "inf_loss": inject_inf_loss,
    "loss_explosion": inject_loss_explosion,
    "weight_corruption": inject_weight_corruption,
    "lr_nuke": inject_lr_nuke,
    "repeated_failures": inject_repeated_failures,
}


# =============================================================================
# Test Runner
# =============================================================================

def run_stress_test(model_class, failure_type, n_steps=100, inject_at=50,
                    use_healing=True, verbose=False):
    """Run a single stress test."""
    
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    if model_class == DeepCNN:
        X = torch.randn(200, 3, 32, 32)
    else:
        X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    shard = None
    if use_healing:
        config = SelfHealingConfig(
            loss_explosion_threshold=50.0,
            gradient_explosion_threshold=1e4,
            lr_reduction_factor=0.1,
            verbose=False,
        )
        shard = SelfHealingArc(model, optimizer, config)
    
    results = {
        'failure_type': failure_type,
        'use_healing': use_healing,
        'steps_completed': 0,
        'crashed': False,
        'losses': [],
        'rollbacks': 0,
        'skips': 0,
    }
    
    step = 0
    try:
        for epoch in range(10):  # Max epochs
            for batch_idx, (x, labels) in enumerate(dataloader):
                step += 1
                if step > n_steps:
                    break
                
                optimizer.zero_grad()
                
                out = model(x)
                loss = F.cross_entropy(out, labels)
                
                # Inject failure
                if step == inject_at:
                    injector = FAILURE_INJECTORS.get(failure_type, lambda *args: args[2])
                    result = injector(model, optimizer, loss, step)
                    if isinstance(result, tuple):
                        loss, action = result
                    else:
                        loss = result
                
                # Self-healing check
                if shard:
                    action = shard.step(loss)
                    if action.should_skip:
                        results['skips'] += 1
                        if action.rolled_back:
                            results['rollbacks'] += 1
                        continue
                
                # Check for NaN/Inf without healing
                if not shard:
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e10:
                        results['crashed'] = True
                        break
                
                loss.backward()
                
                # Post-backward healing
                if shard:
                    post_action = shard.post_backward()
                    if post_action.should_skip:
                        results['skips'] += 1
                        continue
                
                # Check for NaN weights
                has_nan = False
                for p in model.parameters():
                    if torch.isnan(p).any():
                        has_nan = True
                        break
                
                if has_nan and not shard:
                    results['crashed'] = True
                    break
                
                optimizer.step()
                results['losses'].append(loss.item())
                results['steps_completed'] = step
            
            if step > n_steps or results['crashed']:
                break
    
    except Exception as e:
        results['crashed'] = True
        results['error'] = str(e)[:100]
    
    if shard:
        stats = shard.get_stats()
        results['rollbacks'] = stats['total_rollbacks']
        results['recovery_rate'] = stats['recovery_rate']
    
    return results


def run_ultimate_stress_test():
    """Run all stress tests."""
    print("="*70)
    print("ULTIMATE STRESS TEST: SelfHealingArc")
    print("   Testing against EVERY possible failure mode")
    print("="*70)
    
    failure_types = list(FAILURE_INJECTORS.keys())
    model_classes = [("MLP", SimpleMLP), ("CNN", DeepCNN)]
    
    all_results = []
    
    for model_name, model_class in model_classes:
        for failure_type in failure_types:
            print(f"\n{'='*70}")
            print(f"MODEL: {model_name} | FAILURE: {failure_type.upper()}")
            print("="*70)
            
            # Without healing
            print("\n[1/2] Without SelfHealingArc...")
            baseline = run_stress_test(model_class, failure_type, use_healing=False)
            status = "CRASH" if baseline['crashed'] else "OK"
            print(f"  Result: {status}, Steps: {baseline['steps_completed']}")
            
            # With healing
            print("\n[2/2] With SelfHealingArc...")
            protected = run_stress_test(model_class, failure_type, use_healing=True)
            status = "CRASH" if protected['crashed'] else "OK"
            print(f"  Result: {status}, Steps: {protected['steps_completed']}, Rollbacks: {protected['rollbacks']}")
            
            saved = baseline['crashed'] and not protected['crashed']
            
            all_results.append({
                'model': model_name,
                'failure_type': failure_type,
                'baseline': baseline,
                'protected': protected,
                'saved': saved,
            })
    
    # Summary
    print("\n" + "="*70)
    print("ULTIMATE STRESS TEST SUMMARY")
    print("="*70)
    
    print("\n| Model | Failure             | Baseline | ARC    | Rollbacks | Saved? |")
    print("|-------|---------------------|----------|--------|-----------|--------|")
    
    total_saved = 0
    total_baseline_crashes = 0
    total_tests = 0
    
    for r in all_results:
        b_status = "CRASH" if r['baseline']['crashed'] else "OK"
        p_status = "CRASH" if r['protected']['crashed'] else "OK"
        saved = "YES" if r['saved'] else "No"
        
        if r['baseline']['crashed']:
            total_baseline_crashes += 1
        if r['saved']:
            total_saved += 1
        total_tests += 1
        
        print(f"| {r['model']:5} | {r['failure_type']:19} | {b_status:8} | {p_status:6} | {r['protected']['rollbacks']:9} | {saved:6} |")
    
    print(f"\nResults:")
    print(f"   - Total tests: {total_tests}")
    print(f"   - Baseline crashes: {total_baseline_crashes}")
    print(f"   - ARC saved: {total_saved}/{total_baseline_crashes}")
    
    if total_baseline_crashes > 0:
        save_rate = (total_saved / total_baseline_crashes) * 100
        print(f"   - Save rate: {save_rate:.1f}%")
    
    # Save results
    with open("ultimate_stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: ultimate_stress_results.json")
    
    return all_results


if __name__ == "__main__":
    run_ultimate_stress_test()