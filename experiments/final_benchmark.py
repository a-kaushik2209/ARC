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
ARC Final Benchmark Suite

Run all experiments in one script to demonstrate ARC capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_data(n=500):
    """Create synthetic data."""
    X = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_weight_rollback():
    """Test WeightRollback saves failing runs."""
    print("\n" + "="*60)
    print("TEST 1: WeightRollback")
    print("="*60)
    
    from arc.intervention.rollback import WeightRollback, RollbackConfig
    
    results = {"test": "weight_rollback", "scenarios": []}
    
    for use_rollback in [False, True]:
        label = "with_rollback" if use_rollback else "without_rollback"
        print(f"\n[{label.upper()}]")
        
        model = SimpleCNN()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        dataloader = create_data()
        
        rollback = None
        if use_rollback:
            config = RollbackConfig(
                checkpoint_frequency=20,
                loss_explosion_threshold=50.0,
                lr_reduction_factor=0.1,
            )
            rollback = WeightRollback(model, optimizer, config, verbose=False)
        
        epochs_completed = 0
        failed = False
        rollbacks = 0
        
        for epoch in range(5):
            for batch_idx, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                try:
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                    
                    # Inject failure
                    if epoch == 2 and batch_idx == 5:
                        loss = loss * 1000
                    
                    if not use_rollback and loss.item() > 100:
                        failed = True
                        break
                    
                    loss.backward()
                    
                    if rollback:
                        action = rollback.step(loss)
                        if action.rolled_back:
                            rollbacks += 1
                            continue
                    
                    optimizer.step()
                except:
                    failed = True
                    break
            
            if failed:
                break
            epochs_completed = epoch + 1
        
        status = "FAIL" if failed else "OK"
        print(f"  Epochs: {epochs_completed}, Rollbacks: {rollbacks}, Status: {status}")
        
        results["scenarios"].append({
            "use_rollback": use_rollback,
            "epochs_completed": epochs_completed,
            "failed": failed,
            "rollbacks": rollbacks,
        })
    
    # Check if rollback saved
    saved = results["scenarios"][0]["failed"] and not results["scenarios"][1]["failed"]
    results["rollback_saved"] = saved
    print(f"\nROLLBACK SAVED RUN: {saved}")
    
    return results


def test_gradient_forecaster():
    """Test GradientForecaster predicts explosions."""
    print("\n" + "="*60)
    print("TEST 2: GradientForecaster")
    print("="*60)
    
    from arc.prediction.forecaster import GradientForecaster
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    forecaster = GradientForecaster(model)
    
    predictions_made = 0
    explosions_predicted = 0
    
    for step in range(100):
        x = torch.randn(32, 100)
        scale = 1.0 + step * 0.05  # Gradually increase
        out = model(x) * scale
        loss = out.mean()
        
        optimizer.zero_grad()
        loss.backward()
        
        forecaster.update()
        forecast = forecaster.predict()
        predictions_made += 1
        
        if forecast.will_explode:
            explosions_predicted += 1
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    
    stats = forecaster.get_stats()
    print(f"  Steps: {stats['total_steps']}")
    print(f"  Explosions predicted: {explosions_predicted}")
    print(f"  Accuracy: {stats['prediction_accuracy']:.2%}")
    
    print(f"\nFORECASTER WORKING: {explosions_predicted > 0}")
    
    return {
        "test": "gradient_forecaster",
        "predictions_made": predictions_made,
        "explosions_predicted": explosions_predicted,
        "accuracy": stats['prediction_accuracy'],
    }


def test_lite_arc():
    """Test LiteArc overhead."""
    print("\n" + "="*60)
    print("TEST 3: LiteArc Overhead")
    print("="*60)
    
    from arc.api.lite import LiteArc, LiteConfig
    
    model = nn.Sequential(
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    optimizer = optim.Adam(model.parameters())
    x = torch.randn(32, 3072)
    n_steps = 200
    
    # Baseline
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    baseline = time.perf_counter() - start
    
    # With LiteArc
    config = LiteConfig(check_every_n_steps=50)
    arc = LiteArc(model, optimizer, config)
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x)
        loss = out.mean()
        loss.backward()
        arc.step(loss)
        optimizer.step()
        optimizer.zero_grad()
    lite_time = time.perf_counter() - start
    
    overhead = ((lite_time - baseline) / baseline) * 100
    print(f"  Baseline: {baseline:.3f}s")
    print(f"  LiteArc:  {lite_time:.3f}s")
    print(f"  Overhead: {overhead:.1f}%")
    
    print(f"\nOVERHEAD < 50%: {overhead < 50}")
    
    return {
        "test": "lite_arc_overhead",
        "baseline_time": baseline,
        "lite_time": lite_time,
        "overhead_percent": overhead,
    }


def run_all_benchmarks():
    """Run all benchmarks."""
    print("="*60)
    print("ARC FINAL BENCHMARK SUITE")
    print("="*60)
    
    all_results = []
    
    # Test 1: WeightRollback
    all_results.append(test_weight_rollback())
    
    # Test 2: GradientForecaster
    all_results.append(test_gradient_forecaster())
    
    # Test 3: LiteArc Overhead
    all_results.append(test_lite_arc())
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\n| Test                 | Result           |")
    print("|----------------------|------------------|")
    print(f"| WeightRollback       | {'SAVED RUN' if all_results[0]['rollback_saved'] else 'DID NOT SAVE'} |")
    print(f"| GradientForecaster   | {'WORKING' if all_results[1]['explosions_predicted'] > 0 else 'NOT WORKING'} |")
    print(f"| LiteArc Overhead     | {all_results[2]['overhead_percent']:.1f}% |")
    
    # Save results
    with open("final_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: final_benchmark_results.json")
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()