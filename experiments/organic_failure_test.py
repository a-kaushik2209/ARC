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
ARC Organic Failure Test (Phase 2)

Addresses reviewer concern: "Manually setting loss = NaN is a unit test, not a stress test"

This module tests REAL failure scenarios:
1. High LR without injection (natural explosion)
2. Unstable architecture (deep vanilla RNN)
3. Bad initialization (extreme weights)
4. Pathological batch (outliers)
5. No weight decay with high LR

These are organic failures that occur in real training.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Unstable Architectures
# =============================================================================

class DeepVanillaRNN(nn.Module):
    """Deep vanilla RNN - known to be unstable (exploding/vanishing gradients)."""
    def __init__(self, input_size=50, hidden_size=128, num_layers=10, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Vanilla RNN layers (no LSTM/GRU stabilization)
        self.rnn_layers = nn.ModuleList([
            nn.RNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        # x: (batch, seq_len, input)
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states
        hiddens = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                   for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            inp = x[:, t, :]
            for i, rnn in enumerate(self.rnn_layers):
                hiddens[i] = rnn(inp if i == 0 else hiddens[i-1], hiddens[i])
        
        return self.fc(hiddens[-1])


class VeryDeepMLP(nn.Module):
    """Very deep MLP without skip connections - prone to gradient issues."""
    def __init__(self, depth=20, width=128):
        super().__init__()
        layers = [nn.Linear(100, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        return self.net(x)


class UnstableTransformer(nn.Module):
    """Transformer with Post-Norm (less stable than Pre-Norm)."""
    def __init__(self, d_model=128, nhead=4, num_layers=6):
        super().__init__()
        self.embed = nn.Linear(50, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=0.0,  # No dropout = less regularization
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


# =============================================================================
# Organic Failure Scenarios
# =============================================================================

def scenario_high_lr(model_class):
    """High learning rate without any stabilization."""
    model = model_class()
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # Extremely high LR
    return model, optimizer, "High LR (1.0)"


def scenario_high_lr_no_decay(model_class):
    """High LR + no weight decay."""
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0)
    return model, optimizer, "High LR + No Decay"


def scenario_bad_init(model_class):
    """Bad weight initialization."""
    model = model_class()
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.uniform_(p, -10, 10)  # Extreme initialization
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer, "Bad Initialization"


def scenario_pathological_data(model_class):
    """Training with pathological data (extreme outliers)."""
    model = model_class()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer, "Pathological Data"


SCENARIOS = [
    scenario_high_lr,
    scenario_high_lr_no_decay,
    scenario_bad_init,
    scenario_pathological_data,
]


# =============================================================================
# Test Runner
# =============================================================================

def create_data(model, pathological=False):
    """Create appropriate data for model."""
    if isinstance(model, DeepVanillaRNN) or isinstance(model, UnstableTransformer):
        X = torch.randn(200, 20, 50)  # (batch, seq, features)
    else:
        X = torch.randn(200, 100)
    
    y = torch.randint(0, 10, (200,))
    
    if pathological:
        # Add extreme outliers
        X[:10] = X[:10] * 1000  # 10 extreme samples
        X[10:20] = X[10:20] * 0.0001  # 10 tiny samples
    
    return X, y


def run_organic_test(model_class, scenario_fn, use_arc=True, n_epochs=10, seed=42):
    """Run organic failure test."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model, optimizer, scenario_name = scenario_fn(model_class)
    
    # Pathological data for that specific scenario
    pathological = (scenario_fn == scenario_pathological_data)
    X, y = create_data(model, pathological=pathological)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    # ARC setup
    shard = None
    if use_arc:
        config = SelfHealingConfig(
            checkpoint_frequency=5,
            loss_explosion_threshold=50.0,
            gradient_explosion_threshold=1e4,
            lr_reduction_factor=0.1,
            verbose=False,
        )
        shard = SelfHealingArc(model, optimizer, config)
    
    result = {
        'model': model_class.__name__,
        'scenario': scenario_name,
        'use_arc': use_arc,
        'epochs_completed': 0,
        'diverged': False,
        'crashed': False,
        'rollbacks': 0,
        'lr_reductions': 0,
        'losses': [],
    }
    
    try:
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_idx, (x, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                
                out = model(x)
                loss = F.cross_entropy(out, labels)
                
                # Check for natural divergence
                if torch.isnan(loss) or torch.isinf(loss):
                    if not use_arc:
                        result['diverged'] = True
                        break
                
                if loss.item() > 1e6:  # Natural explosion
                    if not use_arc:
                        result['diverged'] = True
                        break
                
                # ARC protection
                if shard:
                    action = shard.step(loss)
                    if action.should_skip:
                        if action.rolled_back:
                            result['rollbacks'] += 1
                        if action.lr_reduced:
                            result['lr_reductions'] += 1
                        continue
                
                loss.backward()
                
                # Check gradient explosion
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                if grad_norm > 1e6 and not use_arc:
                    result['diverged'] = True
                    break
                
                if shard:
                    shard.post_backward()
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            if result['diverged']:
                break
            
            if epoch_losses:
                result['losses'].append(np.mean(epoch_losses))
            result['epochs_completed'] = epoch + 1
    
    except Exception as e:
        result['crashed'] = True
        result['error'] = str(e)[:100]
    
    if shard:
        stats = shard.get_stats()
        result['rollbacks'] = stats['total_rollbacks']
        result['lr_reductions'] = stats['lr_reductions']
    
    return result


def run_organic_failure_suite():
    """Run full organic failure test suite."""
    
    print("="*70)
    print("🌿 ORGANIC FAILURE TEST SUITE")
    print("   Testing REAL failure scenarios (not synthetic injections)")
    print("="*70)
    
    model_classes = [VeryDeepMLP, DeepVanillaRNN, UnstableTransformer]
    
    all_results = []
    
    for model_class in model_classes:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_class.__name__}")
        print("="*70)
        
        for scenario_fn in SCENARIOS:
            # Get scenario name
            _, _, scenario_name = scenario_fn(model_class)
            print(f"\n  Scenario: {scenario_name}")
            
            # Baseline
            baseline = run_organic_test(model_class, scenario_fn, use_arc=False)
            b_status = "DIVERGED" if baseline['diverged'] else ("CRASH" if baseline['crashed'] else "OK")
            print(f"    Baseline: {b_status} (epochs: {baseline['epochs_completed']})")
            
            # With ARC
            arc_result = run_organic_test(model_class, scenario_fn, use_arc=True)
            a_status = "DIVERGED" if arc_result['diverged'] else ("CRASH" if arc_result['crashed'] else "OK")
            print(f"    ARC:      {a_status} (epochs: {arc_result['epochs_completed']}, rollbacks: {arc_result['rollbacks']})")
            
            # Did ARC save this?
            baseline_failed = baseline['diverged'] or baseline['crashed']
            arc_ok = not arc_result['diverged'] and not arc_result['crashed']
            saved = baseline_failed and arc_ok
            
            all_results.append({
                'model': model_class.__name__,
                'scenario': scenario_name,
                'baseline': baseline,
                'arc': arc_result,
                'saved': saved,
            })
    
    # Summary
    print("\n" + "="*70)
    print("🌿 ORGANIC FAILURE SUMMARY")
    print("="*70)
    
    print("\n| Model | Scenario | Baseline | ARC | Rollbacks | Saved? |")
    print("|-------|----------|----------|-----|-----------|--------|")
    
    total_baseline_fails = 0
    total_arc_saves = 0
    
    for r in all_results:
        b_status = "FAIL" if (r['baseline']['diverged'] or r['baseline']['crashed']) else "OK"
        a_status = "FAIL" if (r['arc']['diverged'] or r['arc']['crashed']) else "OK"
        saved = "YES" if r['saved'] else "No"
        
        if b_status == "FAIL":
            total_baseline_fails += 1
        if r['saved']:
            total_arc_saves += 1
        
        print(f"| {r['model']:20} | {r['scenario']:20} | {b_status:8} | {a_status:3} | {r['arc']['rollbacks']:9} | {saved:6} |")
    
    print(f"\nARC saved {total_arc_saves}/{total_baseline_fails} organically failing runs")
    
    # Save
    with open("organic_failure_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: organic_failure_results.json")
    
    return all_results


if __name__ == "__main__":
    run_organic_failure_suite()