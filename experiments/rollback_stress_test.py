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
ARC Phase 14: Rollback Stress Test

This test verifies that the new WeightRollback system actually SAVES failing runs
by actively rolling back and adjusting hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.intervention.rollback import WeightRollback, RollbackConfig


class SimpleCNN(nn.Module):
    """Simple CNN prone to instability."""
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


def create_synthetic_data(n_samples=500):
    """Create synthetic CIFAR-like data."""
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


def train_with_rollback(inject_failure=True, use_rollback=True):
    """Train with optional failure injection and rollback protection."""
    
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    dataloader = create_synthetic_data()
    
    # Setup rollback
    rollback = None
    if use_rollback:
        config = RollbackConfig(
            checkpoint_frequency=20,
            loss_explosion_threshold=50.0,
            gradient_explosion_threshold=1e3,
            lr_reduction_factor=0.1,
        )
        rollback = WeightRollback(model, optimizer, config, verbose=True)
    
    results = {
        "use_rollback": use_rollback,
        "inject_failure": inject_failure,
        "epochs_completed": 0,
        "failed": False,
        "rollbacks_triggered": 0,
        "losses": [],
    }
    
    n_epochs = 5
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            try:
                out = model(x)
                loss = F.cross_entropy(out, y)
                
                # INJECT FAILURE at epoch 2, batch 5
                if inject_failure and epoch == 2 and batch_idx == 5:
                    # Corrupt gradients massively
                    loss = loss * 1000
                    print(f"  [INJECTION] Loss multiplied by 1000x")
                
                # Check for failure without rollback
                if not use_rollback:
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                        print(f"  [FAILURE] Loss exploded: {loss.item()}")
                        results["failed"] = True
                        break
                
                loss.backward()
                
                # Use rollback system
                if rollback:
                    action = rollback.step(loss)
                    if action.rolled_back:
                        results["rollbacks_triggered"] += 1
                        continue  # Skip this bad step
                
                # Check for NaN weights (without rollback)
                if not use_rollback:
                    if any(torch.isnan(p).any() for p in model.parameters()):
                        print(f"  [FAILURE] NaN weights detected")
                        results["failed"] = True
                        break
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
            except RuntimeError as e:
                print(f"  [FAILURE] Runtime: {e}")
                results["failed"] = True
                break
        
        if results["failed"]:
            break
        
        avg_loss = epoch_loss / max(n_batches, 1)
        results["losses"].append(avg_loss)
        results["epochs_completed"] = epoch + 1
        
        if rollback:
            rollback.end_epoch()
        
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return results


def run_rollback_stress_test():
    """Run stress test comparing with/without rollback."""
    
    print("="*60)
    print("ROLLBACK STRESS TEST (Phase 14)")
    print("Testing active intervention system")
    print("="*60)
    
    scenarios = [
        {"inject": False, "rollback": False, "name": "baseline_clean"},
        {"inject": True, "rollback": False, "name": "failure_no_protection"},
        {"inject": True, "rollback": True, "name": "failure_with_rollback"},
    ]
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name'].upper()}")
        print("="*60)
        
        result = train_with_rollback(
            inject_failure=scenario["inject"],
            use_rollback=scenario["rollback"]
        )
        
        status = "FAILED" if result["failed"] else f"OK (Rollbacks: {result['rollbacks_triggered']})"
        print(f"  Final Result: {status}")
        
        all_results.append({
            "scenario": scenario["name"],
            "result": result,
        })
    
    # Summary
    print("\n" + "="*60)
    print("ROLLBACK STRESS TEST SUMMARY")
    print("="*60)
    
    print("\n| Scenario               | Status    | Rollbacks | Epochs |")
    print("|------------------------|-----------|-----------|--------|")
    
    rollback_saved = False
    for r in all_results:
        status = "FAIL" if r["result"]["failed"] else "OK"
        rollbacks = r["result"]["rollbacks_triggered"]
        epochs = r["result"]["epochs_completed"]
        print(f"| {r['scenario']:22} | {status:9} | {rollbacks:9} | {epochs:6} |")
        
        if r["scenario"] == "failure_with_rollback" and not r["result"]["failed"]:
            rollback_saved = True
    
    if rollback_saved:
        print("\nROLLBACK SAVED THE FAILING RUN!")
    else:
        print("\nRollback did not save run (may need tuning)")
    
    # Save results
    with open("rollback_stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: rollback_stress_results.json")
    
    return all_results


if __name__ == "__main__":
    run_rollback_stress_test()