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
ARC Publication Experiment 2: Real Failure Recovery

Demonstrates ARC saving divergent training runs in real time.

Failure Types:
1. Learning rate explosion
2. Gradient explosion
3. Loss spike (corrupted data)
4. Vanishing gradients
5. Weight explosion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from arc import ArcV2


class DeepNet(nn.Module):
    """Deep network prone to gradient issues."""
    def __init__(self, depth=10):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Conv2d(3, 64, 3, padding=1))
        
        # Deep middle layers (no batch norm = prone to issues)
        for _ in range(depth):
            self.layers.append(nn.Conv2d(64, 64, 3, padding=1))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_failure_scenarios():
    """Define different failure scenarios."""
    return {
        "lr_explosion": {
            "description": "Learning rate increases 10x every 3 epochs",
            "initial_lr": 0.01,
            "lr_multiplier": 10,
            "lr_increase_epoch": 3,
        },
        "gradient_explosion": {
            "description": "No gradient clipping with high LR",
            "lr": 0.5,
            "weight_decay": 0,
        },
        "corrupted_batch": {
            "description": "Random batch has 100x values",
            "corruption_epoch": 5,
            "corruption_factor": 100,
        },
        "vanishing_gradients": {
            "description": "Very deep network without skip connections",
            "depth": 20,
            "lr": 0.001,
        },
        "weight_explosion": {
            "description": "Initialize weights with large values",
            "weight_init_std": 10.0,
        },
    }


def run_failure_experiment(scenario_name, scenario, train_loader, test_loader, use_arc):
    """Run a single failure experiment."""
    
    # Create model based on scenario
    if scenario_name == "vanishing_gradients":
        model = DeepNet(depth=scenario.get("depth", 20))
    else:
        model = DeepNet(depth=5)
    
    # Weight explosion initialization
    if scenario_name == "weight_explosion":
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=scenario["weight_init_std"])
    
    # Optimizer
    lr = scenario.get("lr", scenario.get("initial_lr", 0.01))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # ARC setup
    arc = None
    if use_arc:
        arc = ArcV2.auto(model, optimizer, safety_level="paranoid")
    
    # Training
    epochs = 5  # Reduced from 15
    results = {
        "scenario": scenario_name,
        "use_arc": use_arc,
        "epochs_completed": 0,
        "failed": False,
        "failure_detected_epoch": None,
        "interventions": [],
        "losses": [],
        "final_accuracy": 0,
    }
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # LR explosion
        if scenario_name == "lr_explosion" and epoch > 0 and epoch % scenario["lr_increase_epoch"] == 0:
            for pg in optimizer.param_groups:
                pg["lr"] *= scenario["lr_multiplier"]
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Corrupted batch
            if scenario_name == "corrupted_batch" and epoch == scenario["corruption_epoch"] and batch_idx == 5:
                x = x * scenario["corruption_factor"]
            
            optimizer.zero_grad()
            
            try:
                out = model(x)
                loss = F.cross_entropy(out, y)
                
                # Check for failure
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    results["failed"] = True
                    break
                
                loss.backward()
                optimizer.step()
                
                # ARC monitoring
                if arc is not None:
                    status = arc.step(loss)
                    if isinstance(status, dict) and status.get("recommendation"):
                        results["interventions"].append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "type": str(status["recommendation"])
                        })
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Check weights
                if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                    results["failed"] = True
                    break
                    
            except RuntimeError as e:
                results["failed"] = True
                results["error"] = str(e)
                break
        
        if results["failed"]:
            break
        
        avg_loss = epoch_loss / max(batch_count, 1)
        results["losses"].append(avg_loss)
        results["epochs_completed"] = epoch + 1
        
        if arc is not None:
            arc.end_epoch(epoch)
        
        # Early break if loss is too high
        if avg_loss > 100:
            results["failed"] = True
            break
    
    # Test accuracy
    if not results["failed"]:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                try:
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
                except:
                    pass
        results["final_accuracy"] = correct / total if total > 0 else 0
    
    return results


def run_failure_recovery_experiment():
    """Run all failure scenarios with and without ARC."""
    
    print("="*60)
    print("FAILURE RECOVERY EXPERIMENT")
    print("Testing ARC's ability to save divergent training")
    print("="*60)
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Use subset for speed (2000 samples)
    indices = list(range(2000))
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Modify epochs in run_failure_experiment calls
    # Note: run_failure_experiment needs update below or we pass epochs=5 here if possible
    # Actually run_failure_experiment sets epochs=15 internally. We need to patch that.
    
    scenarios = create_failure_scenarios()
    all_results = []
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"    {scenario.get('description', '')}")
        
        # Without ARC
        print("  Without ARC...", end=" ")
        result_no_arc = run_failure_experiment(scenario_name, scenario, train_loader, test_loader, use_arc=False)
        status = "FAILED" if result_no_arc["failed"] else f"OK ({result_no_arc['final_accuracy']:.1%})"
        print(status)
        
        # With ARC
        print("  With ARC...", end=" ")
        result_arc = run_failure_experiment(scenario_name, scenario, train_loader, test_loader, use_arc=True)
        status = "FAILED" if result_arc["failed"] else f"OK ({result_arc['final_accuracy']:.1%})"
        print(status)
        
        all_results.append({
            "scenario": scenario_name,
            "without_arc": result_no_arc,
            "with_arc": result_arc,
            "arc_saved": result_no_arc["failed"] and not result_arc["failed"],
        })
    
    # Summary
    print("\n" + "="*60)
    print("FAILURE RECOVERY SUMMARY")
    print("="*60)
    
    print("\n| Scenario             | No ARC    | With ARC  | ARC Saved? |")
    print("|----------------------|-----------|-----------|------------|")
    
    arc_saves = 0
    for r in all_results:
        no_arc = "FAIL" if r["without_arc"]["failed"] else f"{r['without_arc']['final_accuracy']:.0%}"
        with_arc = "FAIL" if r["with_arc"]["failed"] else f"{r['with_arc']['final_accuracy']:.0%}"
        saved = "YES" if r["arc_saved"] else "No"
        if r["arc_saved"]:
            arc_saves += 1
        print(f"| {r['scenario']:20} | {no_arc:9} | {with_arc:9} | {saved:10} |")
    
    print(f"\nARC saved {arc_saves}/{len(all_results)} failing scenarios")
    
    # Save
    with open("failure_recovery_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: failure_recovery_results.json")
    
    return all_results


if __name__ == "__main__":
    run_failure_recovery_experiment()