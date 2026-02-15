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
ARC v2.0 Real Dataset Scientific Benchmark

Uses REAL MNIST and CIFAR-10 datasets for publication-ready results.

How to run:
1. pip install torchvision
2. python experiments/real_data_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for torchvision
try:
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST, CIFAR10
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Please install torchvision: pip install torchvision")
    sys.exit(1)

from arc import ElasticWeightConsolidation, ConformalPredictor


# =============================================================================
# MODELS
# =============================================================================

class MNISTClassifier(nn.Module):
    """CNN for MNIST."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CIFAR10Classifier(nn.Module):
    """CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# BENCHMARK 1: SPLIT MNIST (Continual Learning)
# =============================================================================

def benchmark_split_mnist_real():
    """
    Split MNIST Benchmark using REAL MNIST data.
    
    Tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
    """
    print("\n" + "="*60)
    print("BENCHMARK 1: SPLIT MNIST (Real Data)")
    print("="*60)
    
    # Download MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("  Downloading MNIST...")
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data', train=False, download=True, transform=transform)
    
    n_tasks = 5
    results = {
        "baseline": {"task_acc": [], "avg_acc": 0, "bwt": 0},
        "ewc": {"task_acc": [], "avg_acc": 0, "bwt": 0},
    }
    
    for method_name in ["baseline", "ewc"]:
        print(f"\n  Testing: {method_name.upper()}")
        
        model = MNISTClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        ewc = None
        if method_name == "ewc":
            ewc = ElasticWeightConsolidation(model, lambda_ewc=5000, online=True)
        
        # Track accuracy after training each task
        accuracy_after_task = np.zeros((n_tasks, n_tasks))
        
        for task_id in range(n_tasks):
            digit1 = task_id * 2
            digit2 = task_id * 2 + 1
            
            print(f"    Task {task_id+1}/5: Digits {digit1} & {digit2}")
            
            # Filter data for this task
            train_indices = [i for i, (_, label) in enumerate(train_dataset) 
                           if label in [digit1, digit2]]
            task_train = Subset(train_dataset, train_indices[:2000])
            
            train_loader = DataLoader(task_train, batch_size=64, shuffle=True)
            
            # Train
            model.train()
            for epoch in range(5):
                for x, y in train_loader:
                    # Binary labels for this task
                    y_binary = (y == digit2).long()
                    
                    optimizer.zero_grad()
                    
                    # Modify model output for binary classification
                    out = model(x)[:, :2]  # Use first 2 outputs
                    loss = F.cross_entropy(out, y_binary)
                    
                    if ewc is not None and ewc.n_tasks > 0:
                        loss = loss + ewc.compute_penalty()
                    
                    loss.backward()
                    optimizer.step()
            
            # Consolidate
            if ewc is not None:
                ewc.consolidate_task(f"task_{task_id}", train_loader)
            
            # Evaluate on all tasks seen so far
            model.eval()
            for eval_task_id in range(task_id + 1):
                eval_d1 = eval_task_id * 2
                eval_d2 = eval_task_id * 2 + 1
                
                test_indices = [i for i, (_, label) in enumerate(test_dataset)
                              if label in [eval_d1, eval_d2]]
                task_test = Subset(test_dataset, test_indices[:500])
                test_loader = DataLoader(task_test, batch_size=64)
                
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        y_binary = (y == eval_d2).long()
                        out = model(x)[:, :2]
                        pred = out.argmax(dim=1)
                        correct += (pred == y_binary).sum().item()
                        total += len(y)
                
                accuracy_after_task[task_id, eval_task_id] = correct / total
        
        # Compute metrics
        avg_acc = accuracy_after_task[-1, :].mean()
        
        bwt = 0
        for j in range(n_tasks - 1):
            bwt += accuracy_after_task[-1, j] - accuracy_after_task[j, j]
        bwt /= (n_tasks - 1)
        
        results[method_name]["task_acc"] = accuracy_after_task[-1, :].tolist()
        results[method_name]["avg_acc"] = avg_acc
        results[method_name]["bwt"] = bwt
        
        print(f"      Average Accuracy: {avg_acc:.1%}")
        print(f"      Backward Transfer: {bwt:+.3f}")
    
    return results


# =============================================================================
# BENCHMARK 2: CIFAR-10 CONFORMAL PREDICTION
# =============================================================================

def benchmark_cifar10_conformal():
    """
    Conformal Prediction on REAL CIFAR-10.
    
    Tests coverage guarantees with real data.
    """
    print("\n" + "="*60)
    print("BENCHMARK 2: CIFAR-10 CONFORMAL PREDICTION (Real Data)")
    print("="*60)
    
    # Download CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print("  Downloading CIFAR-10...")
    train_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Split: train (40k), calibration (10k), test (10k)
    train_size = 40000
    cal_size = 10000
    
    train_set, cal_set = random_split(train_dataset, [train_size, cal_size])
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    cal_loader = DataLoader(cal_set, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Train model
    print("  Training CIFAR-10 classifier...")
    model = CIFAR10Classifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"    Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate base accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    base_accuracy = correct / total
    print(f"  Base Test Accuracy: {base_accuracy:.1%}")
    
    # Conformal prediction at different alpha levels
    results = {"base_accuracy": base_accuracy}
    
    for alpha in [0.10, 0.05, 0.01]:
        print(f"\n  Testing alpha = {alpha}...")
        
        cp = ConformalPredictor(model, alpha=alpha)
        cp.calibrate(cal_loader)
        
        # Evaluate on test set
        covered = 0
        total_size = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                for i in range(len(x)):
                    pred_set = cp.predict(x[i:i+1])
                    if y[i].item() in pred_set.set_members:
                        covered += 1
                    total_size += pred_set.set_size
                    total_samples += 1
        
        empirical_cov = covered / total_samples
        avg_set_size = total_size / total_samples
        
        results[f"alpha_{alpha}"] = {
            "target": 1 - alpha,
            "empirical": empirical_cov,
            "set_size": avg_set_size,
            "valid": empirical_cov >= (1 - alpha) - 0.02
        }
        
        status = "PASS" if empirical_cov >= (1 - alpha) - 0.02 else "FAIL"
        print(f"    Target: {1-alpha:.0%}, Actual: {empirical_cov:.1%}")
        print(f"    Avg Set Size: {avg_set_size:.2f}")
        print(f"    Status: {status}")
    
    return results


# =============================================================================
# BENCHMARK 3: MNIST FAILURE DETECTION
# =============================================================================

def benchmark_mnist_failure_detection():
    """
    Test failure detection on MNIST with induced failures.
    """
    print("\n" + "="*60)
    print("BENCHMARK 3: MNIST FAILURE DETECTION (Real Data)")
    print("="*60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    n_healthy = 10
    n_failures = 10
    
    predictions = []
    ground_truth = []
    
    print("  Running healthy training runs...")
    for run in range(n_healthy):
        model = MNISTClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        losses = []
        model.train()
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            batch_count += 1
            if batch_count >= 100:
                break
        
        risk = np.var(losses[-20:]) / 10 + (1 if max(losses) > 10 else 0)
        predictions.append(min(risk, 1.0))
        ground_truth.append(0)
    
    print("  Running failure training runs...")
    for run in range(n_failures):
        model = MNISTClassifier()
        optimizer = optim.Adam(model.parameters(), lr=0.5)  # Too high LR!
        
        losses = []
        model.train()
        batch_count = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            batch_count += 1
            if batch_count >= 100:
                break
        
        risk = np.var(losses[-20:]) / 10 + (1 if max(losses) > 10 else 0)
        predictions.append(min(risk, 1.0))
        ground_truth.append(1)
    
    # Metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    threshold = 0.5
    pred_labels = (predictions > threshold).astype(int)
    
    tp = ((pred_labels == 1) & (ground_truth == 1)).sum()
    fp = ((pred_labels == 1) & (ground_truth == 0)).sum()
    fn = ((pred_labels == 0) & (ground_truth == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "healthy_avg_risk": predictions[ground_truth == 0].mean(),
        "failure_avg_risk": predictions[ground_truth == 1].mean(),
    }
    
    print(f"\n  Results:")
    print(f"    Precision: {precision:.1%}")
    print(f"    Recall: {recall:.1%}")
    print(f"    F1 Score: {f1:.1%}")
    print(f"    Healthy Avg Risk: {results['healthy_avg_risk']:.3f}")
    print(f"    Failure Avg Risk: {results['failure_avg_risk']:.3f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("ARC v2.0 REAL DATA BENCHMARK")
    print("MNIST + CIFAR-10 Scientific Evaluation")
    print("="*60)
    
    start_time = time.time()
    
    all_results = {}
    
    # Run benchmarks
    all_results["split_mnist"] = benchmark_split_mnist_real()
    all_results["cifar10_conformal"] = benchmark_cifar10_conformal()
    all_results["mnist_failure"] = benchmark_mnist_failure_detection()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("REAL DATA BENCHMARK SUMMARY")
    print("="*60)
    
    print("\n1. SPLIT MNIST (Continual Learning)")
    print("   | Method   | Avg Acc | BWT    |")
    print("   |----------|---------|--------|")
    for method in ["baseline", "ewc"]:
        r = all_results["split_mnist"][method]
        print(f"   | {method:8} | {r['avg_acc']:.1%}  | {r['bwt']:+.3f} |")
    
    print("\n2. CIFAR-10 CONFORMAL")
    print(f"   Base Accuracy: {all_results['cifar10_conformal']['base_accuracy']:.1%}")
    print("   | Alpha | Target | Actual | Set Size |")
    print("   |-------|--------|--------|----------|")
    for alpha in [0.10, 0.05, 0.01]:
        r = all_results['cifar10_conformal'][f'alpha_{alpha}']
        print(f"   | {alpha:.2f}  | {r['target']:.0%}   | {r['empirical']:.1%}  | {r['set_size']:.2f}     |")
    
    print("\n3. MNIST FAILURE DETECTION")
    r = all_results["mnist_failure"]
    print(f"   Precision: {r['precision']:.1%}")
    print(f"   Recall: {r['recall']:.1%}")
    print(f"   F1: {r['f1']:.1%}")
    
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save results
    with open("real_data_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: real_data_benchmark_results.json")
    
    return all_results


if __name__ == "__main__":
    main()