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
ARC Extended Statistical Validation (30 Seeds)

Addresses reviewer concern: "n=10 is informative but small"

This module runs 30-seed validation for stronger statistical claims.
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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# Extended seed list (30 seeds)
SEEDS = [
    42, 123, 456, 789, 1337, 2024, 3141, 5926, 8675, 3099,  # Original 10
    1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010,  # 10 more
    2020, 3030, 4040, 5050, 6060, 7070, 8080, 9090, 1212, 3434,  # 10 more
]

N_STEPS = 100
INJECT_AT = 50


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


def run_single_test(failure_type, seed, use_arc=True):
    """Run a single test with specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    shard = None
    if use_arc:
        config = SelfHealingConfig(
            checkpoint_frequency=10,
            loss_explosion_threshold=100.0,
            verbose=False,
        )
        shard = SelfHealingArc(model, optimizer, config)
    
    result = {'seed': seed, 'recovered': False, 'crashed': False, 'rollbacks': 0}
    
    step = 0
    try:
        for epoch in range(10):
            for batch_idx, (x, labels) in enumerate(dataloader):
                step += 1
                if step > N_STEPS:
                    break
                
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, labels)
                
                # Inject failure
                if step == INJECT_AT:
                    if failure_type == "nan":
                        loss = torch.tensor(float('nan'))
                    elif failure_type == "inf":
                        loss = torch.tensor(float('inf'))
                
                if shard:
                    action = shard.step(loss)
                    if action.should_skip:
                        if action.rolled_back:
                            result['rollbacks'] += 1
                        continue
                else:
                    if torch.isnan(loss) or torch.isinf(loss):
                        result['crashed'] = True
                        break
                
                loss.backward()
                if shard:
                    shard.post_backward()
                optimizer.step()
            
            if step > N_STEPS or result['crashed']:
                break
    except:
        result['crashed'] = True
    
    result['recovered'] = not result['crashed'] and step >= N_STEPS
    return result


def run_30_seed_validation():
    """Run 30-seed validation."""
    
    print("="*70)
    print("30-SEED STATISTICAL VALIDATION")
    print(f"   {len(SEEDS)} seeds × 2 failures = {len(SEEDS)*2} runs")
    print("="*70)
    
    results = {}
    
    for failure_type in ["nan", "inf"]:
        print(f"\n  Failure: {failure_type.upper()}")
        
        recoveries = []
        rollbacks = []
        
        for i, seed in enumerate(SEEDS):
            result = run_single_test(failure_type, seed, use_arc=True)
            recoveries.append(1 if result['recovered'] else 0)
            rollbacks.append(result['rollbacks'])
            
            status = "✅" if result['recovered'] else "❌"
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(SEEDS)} seeds completed")
        
        # Statistics
        n = len(SEEDS)
        mean = np.mean(recoveries)
        std = np.std(recoveries, ddof=1) if n > 1 else 0
        
        # 95% CI (t-distribution)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n)) if std > 0 else (mean, mean)
        ci_95 = (max(0, ci_95[0]), min(1, ci_95[1]))
        
        # Bootstrap p-value
        boot_means = []
        for _ in range(10000):
            sample = np.random.choice(recoveries, size=n, replace=True)
            boot_means.append(np.mean(sample))
        p_value = np.mean(np.array(boot_means) <= 0)
        
        results[failure_type] = {
            'n_seeds': n,
            'recovery_rate': mean,
            'std': std,
            'ci_95_low': ci_95[0],
            'ci_95_high': ci_95[1],
            'p_value': p_value,
            'mean_rollbacks': np.mean(rollbacks),
        }
        
        print(f"    Recovery: {mean*100:.1f}% ± {std*100:.1f}%")
        print(f"    95% CI: [{ci_95[0]*100:.1f}%, {ci_95[1]*100:.1f}%]")
        print(f"    p-value: {p_value:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("30-SEED VALIDATION SUMMARY")
    print("="*70)
    
    print("\n| Failure | N | Recovery | 95% CI | p-value |")
    print("|---------|---|----------|--------|---------|")
    
    for failure, r in results.items():
        print(f"| {failure:7} | {r['n_seeds']} | {r['recovery_rate']*100:.1f}% ± {r['std']*100:.1f}% | [{r['ci_95_low']*100:.0f}%, {r['ci_95_high']*100:.0f}%] | {r['p_value']:.6f} |")
    
    # Save
    with open("30_seed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\nResults saved to: 30_seed_results.json")
    
    return results


if __name__ == "__main__":
    run_30_seed_validation()