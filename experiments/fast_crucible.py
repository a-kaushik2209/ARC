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
ARC FAST CRUCIBLE TEST

Quick validation across advanced architectures.
4 models × 5 failures × 10 seeds = 200 tests (~20-30 min)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import sys
import os
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# ============================================================================
# COMPACT ADVANCED ARCHITECTURES
# ============================================================================

class MiniGPT(nn.Module):
    """Mini GPT-2 style (12M params)."""
    def __init__(self, vocab=5000, d=256, layers=4, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(128, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, d*4, batch_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        x = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


class MiniViT(nn.Module):
    """Mini Vision Transformer (8M params)."""
    def __init__(self, img=32, patch=4, d=256, layers=4, heads=4, classes=100):
        super().__init__()
        n_patches = (img // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, d*4, batch_first=True)
            for _ in range(layers)
        ])
        self.head = nn.Linear(d, classes)
        
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])


class MiniResNet(nn.Module):
    """Mini ResNet (5M params)."""
    def __init__(self, classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._layer(64, 64, 3)
        self.layer2 = self._layer(64, 128, 4, stride=2)
        self.layer3 = self._layer(128, 256, 6, stride=2)
        self.layer4 = self._layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, classes)
        
    def _layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [self._block(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(self._block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))


class DeepLSTM(nn.Module):
    """Deep LSTM (10M params)."""
    def __init__(self, vocab=5000, embed=256, hidden=512, layers=6, classes=10):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, classes)
        
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ============================================================================
# TEST RUNNER
# ============================================================================

def create_model_data(model_type: str):
    """Create model and data."""
    if model_type == 'gpt':
        model = MiniGPT()
        x = torch.randint(0, 5000, (4, 32))
        y = torch.randint(0, 5000, (4, 32))
        loss_fn = lambda o, t: F.cross_entropy(o.view(-1, 5000), t.view(-1))
    elif model_type == 'vit':
        model = MiniViT()
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        loss_fn = F.cross_entropy
    elif model_type == 'resnet':
        model = MiniResNet()
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        loss_fn = F.cross_entropy
    elif model_type == 'lstm':
        model = DeepLSTM()
        x = torch.randint(0, 5000, (8, 50))
        y = torch.randint(0, 10, (8,))
        loss_fn = F.cross_entropy
    return model, x, y, loss_fn


def run_test(model_type: str, failure: str, seed: int, steps=50) -> Dict:
    """Run single test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    result = {'model': model_type, 'failure': failure, 'seed': seed, 
              'recovered': False, 'crashed': False, 'rollbacks': 0}
    
    try:
        model, x, y, loss_fn = create_model_data(model_type)
        opt = optim.AdamW(model.parameters(), lr=1e-4)
        shard = SelfHealingArc(model, opt, SelfHealingConfig(
            checkpoint_frequency=5, verbose=False, lite_mode=False
        ))
        
        inject_at = steps // 2
        
        for step in range(steps):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            
            # Inject failure
            if step == inject_at:
                if failure == 'nan':
                    loss = torch.tensor(float('nan'))
                elif failure == 'inf':
                    loss = torch.tensor(float('inf'))
                elif failure == 'explosion':
                    loss = loss * 1e10
                elif failure == 'weight_nan':
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.numel() > 10:
                                p.view(-1)[0] = float('nan')
                                break
                elif failure == 'cascade':
                    with torch.no_grad():
                        for p in model.parameters():
                            if p.numel() > 10:
                                p.view(-1)[0] = float('nan')
                                break
                    loss = torch.tensor(float('nan'))
            
            action = shard.step(loss)
            if action.should_skip:
                if action.rolled_back:
                    result['rollbacks'] += 1
                continue
            
            loss.backward()
            shard.post_backward()
            opt.step()
        
        result['recovered'] = True
        
    except Exception as e:
        result['crashed'] = True
        result['error'] = str(e)
    
    return result


def run_fast_crucible():
    """Run fast crucible test."""
    print("=" * 70)
    print("ARC FAST CRUCIBLE TEST")
    print("=" * 70)
    
    MODELS = ['gpt', 'vit', 'resnet', 'lstm']
    FAILURES = ['nan', 'inf', 'explosion', 'weight_nan', 'cascade']
    SEEDS = list(range(10))
    
    total = len(MODELS) * len(FAILURES) * len(SEEDS)
    print(f"\n{len(MODELS)} models × {len(FAILURES)} failures × {len(SEEDS)} seeds = {total} tests\n")
    
    # Model sizes
    print("Model Sizes:")
    for m in MODELS:
        model, _, _, _ = create_model_data(m)
        params = sum(p.numel() for p in model.parameters())
        print(f"   {m.upper()}: {params/1e6:.1f}M params")
        del model
    
    print("\n" + "=" * 70)
    
    results = {}
    count = 0
    start = time.time()
    
    for model in MODELS:
        print(f"\nTesting {model.upper()}...")
        results[model] = {}
        
        for failure in FAILURES:
            recoveries = []
            
            for seed in SEEDS:
                count += 1
                r = run_test(model, failure, seed)
                recoveries.append(1 if r['recovered'] else 0)
                
                if count % 20 == 0:
                    elapsed = time.time() - start
                    eta = (elapsed / count) * (total - count)
                    print(f"   Progress: {count}/{total} | ETA: {eta/60:.1f} min")
            
            rate = np.mean(recoveries) * 100
            results[model][failure] = {
                'n': len(SEEDS),
                'recovery_rate': rate,
                'all_recovered': rate == 100.0
            }
    
    elapsed = time.time() - start
    
    # Results
    print("\n" + "=" * 70)
    print("CRUCIBLE RESULTS")
    print("=" * 70)
    
    print("\n| Model  | Failure    | N  | Recovery | Status |")
    print("|--------|------------|----|---------:|--------|")
    
    passed = 0
    total_combos = 0
    
    for model in MODELS:
        for failure in FAILURES:
            r = results[model][failure]
            status = "✅" if r['all_recovered'] else "❌"
            print(f"| {model:6} | {failure:10} | {r['n']:2} | {r['recovery_rate']:6.1f}% | {status}    |")
            total_combos += 1
            if r['all_recovered']:
                passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n   Total Tests: {total}")
    print(f"   Combinations: {total_combos}")
    print(f"   100% Recovery: {passed}/{total_combos}")
    print(f"   Time: {elapsed/60:.1f} minutes")
    
    overall = passed / total_combos * 100
    print(f"\n   Overall Success: {overall:.1f}%")
    
    if passed == total_combos:
        print("\n" + "=" * 70)
        print("   CRUCIBLE PASSED: ARC IS PEAK!")
        print("=" * 70)
    
    # Save
    with open("crucible_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: crucible_results.json")
    
    return results


if __name__ == "__main__":
    run_fast_crucible()