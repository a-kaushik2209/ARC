"""
Expanded Prediction Experiment (200+ scenarios)
Tests ARC's failure prediction across diverse conditions:
  - 4 architectures × 5 failure types × 5 seeds × 2 labels (failure/stable) = 200 scenarios
  - Train logistic classifier on 80%, test on 20%
  - Report accuracy, precision, recall, F1 with confidence intervals
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import copy
import math
import time
import os
import sys

DEVICE = "cpu"

# ─── Helpers ───────────────────────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)

def get_cifar10_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

# ─── Architectures ─────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*8*8, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MLP(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 10)
        )
    def forward(self, x):
        return self.net(x)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class WideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

ARCHITECTURES = {
    'TinyCNN': TinyCNN,
    'MLP': lambda: MLP(256),
    'DeepCNN': DeepCNN,
    'WideCNN': WideCNN,
}

# ─── Failure Modes ─────────────────────────────────────────────────────────
def inject_lr_spike(model, optimizer):
    for pg in optimizer.param_groups:
        pg['lr'] = 5.0

def inject_weight_noise(model, optimizer):
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * 0.3

def inject_grad_zeroing(model, optimizer):
    """Zero out all gradients permanently by corrupting weights"""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 100.0  # scale up to cause gradient issues

def inject_nan(model, optimizer):
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p.data) < 0.1
            p.data[mask] = float('nan')

def inject_momentum_corruption(model, optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                v.mul_(100.0)

FAILURES = {
    'lr_spike': inject_lr_spike,
    'weight_noise': inject_weight_noise,
    'grad_zeroing': inject_grad_zeroing,
    'nan_injection': inject_nan,
    'momentum_corruption': inject_momentum_corruption,
}

# ─── Feature Extraction ───────────────────────────────────────────────────
def collect_signal_features(model, optimizer, criterion, loader_iter, num_steps=30,
                            inject_failure=False, failure_fn=None, inject_at=15):
    """
    Train for num_steps and collect signal features.
    If inject_failure, apply failure_fn at step inject_at.
    Returns feature vector: [loss_trend, loss_var, grad_mean, grad_var, grad_max,
                             weight_norm_change, weight_var, nan_count]
    """
    losses = []
    grad_norms = []
    weight_norms = []
    nan_count = 0
    
    initial_wnorm = 0.0
    for p in model.parameters():
        initial_wnorm += p.data.norm(2).item() ** 2
    initial_wnorm = math.sqrt(initial_wnorm)
    
    for step in range(num_steps):
        if inject_failure and step == inject_at and failure_fn is not None:
            failure_fn(model, optimizer)
        
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        try:
            out = model(x)
            loss = criterion(out, y)
            loss_val = loss.item()
            
            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_count += 1
                losses.append(100.0)  # sentinel
                grad_norms.append(100.0)
                continue
            
            losses.append(loss_val)
            loss.backward()
            
            # Grad norm
            total_gnorm = 0.0
            has_nan = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any():
                        has_nan = True
                        nan_count += 1
                    total_gnorm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(math.sqrt(total_gnorm))
            
            if not has_nan:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        
        except RuntimeError:
            nan_count += 1
            losses.append(100.0)
            grad_norms.append(100.0)
        
        # Weight norm
        curr_wnorm = 0.0
        for p in model.parameters():
            curr_wnorm += p.data.norm(2).item() ** 2
        weight_norms.append(math.sqrt(curr_wnorm))
    
    # Compute features
    if len(losses) < 2:
        return [0]*8
    
    losses = np.array(losses)
    grad_norms = np.array(grad_norms)
    weight_norms = np.array(weight_norms)
    
    # Loss features
    loss_trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)
    loss_var = np.var(losses)
    
    # Gradient features
    grad_mean = np.mean(grad_norms)
    grad_var = np.var(grad_norms)
    grad_max = np.max(grad_norms)
    
    # Weight features
    weight_norm_change = (weight_norms[-1] - initial_wnorm) / max(abs(initial_wnorm), 1e-8)
    weight_var = np.var(weight_norms)
    
    features = [loss_trend, loss_var, grad_mean, grad_var, grad_max,
                weight_norm_change, weight_var, float(nan_count)]
    
    # Replace any NaN/Inf in features
    features = [0.0 if (math.isnan(f) or math.isinf(f)) else f for f in features]
    
    return features


# ─── Logistic Regression ──────────────────────────────────────────────────
class LogisticRegression:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        for _ in range(epochs):
            p = self.predict_proba(X)
            error = p - y
            self.weights -= lr * (X.T @ error) / len(y)
            self.bias -= lr * np.mean(error)


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  EXPANDED PREDICTION EXPERIMENT (200+ Scenarios)")
    print("=" * 70)
    
    loader = get_cifar10_loader(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    seeds = [42, 43, 44, 45, 46]
    
    all_features = []
    all_labels = []  # 1 = failure, 0 = stable
    scenario_info = []
    
    total = len(ARCHITECTURES) * len(FAILURES) * len(seeds) * 2  # ×2 for stable + failure
    count = 0
    
    for arch_name, arch_fn in ARCHITECTURES.items():
        for fail_name, fail_fn in FAILURES.items():
            for seed in seeds:
                # ── Failure scenario ──
                count += 1
                set_seed(seed)
                model = arch_fn().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                features = collect_signal_features(
                    model, optimizer, criterion, loader_iter,
                    num_steps=30, inject_failure=True, failure_fn=fail_fn, inject_at=15
                )
                all_features.append(features)
                all_labels.append(1)
                scenario_info.append(f"{arch_name}/{fail_name}/seed{seed}/FAILURE")
                
                print(f"  [{count:3d}/{total}] {arch_name}/{fail_name} seed={seed} FAILURE  features={[f'{f:.2f}' for f in features[:3]]}")
                
                # ── Stable scenario ──  
                count += 1
                set_seed(seed)
                model = arch_fn().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                features = collect_signal_features(
                    model, optimizer, criterion, loader_iter,
                    num_steps=30, inject_failure=False
                )
                all_features.append(features)
                all_labels.append(0)
                scenario_info.append(f"{arch_name}/{fail_name}/seed{seed}/STABLE")
                
                print(f"  [{count:3d}/{total}] {arch_name}/{fail_name} seed={seed} STABLE   features={[f'{f:.2f}' for f in features[:3]]}")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n  Total scenarios: {len(y)} ({sum(y)} failure, {len(y) - sum(y)} stable)")
    
    # ── Normalise features ──
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    
    # ── 5-fold cross-validation ──
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // 5
    
    all_preds = np.zeros(n)
    all_true = np.zeros(n)
    
    fold_metrics = []
    
    for fold in range(5):
        test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_train, y_train = X_norm[train_idx], y[train_idx]
        X_test, y_test = X_norm[test_idx], y[test_idx]
        
        clf = LogisticRegression(X_train.shape[1])
        clf.fit(X_train, y_train, lr=0.1, epochs=2000)
        
        preds = clf.predict(X_test)
        all_preds[test_idx] = preds
        all_true[test_idx] = y_test
        
        # Fold metrics
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))
        
        acc = (tp + tn) / len(y_test)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        
        fold_metrics.append({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
        print(f"\n  Fold {fold+1}: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")
    
    # ── Overall metrics ──
    tp = np.sum((all_preds == 1) & (all_true == 1))
    fp = np.sum((all_preds == 1) & (all_true == 0))
    fn = np.sum((all_preds == 0) & (all_true == 1))
    tn = np.sum((all_preds == 0) & (all_true == 0))
    
    accuracy = (tp + tn) / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Confidence intervals from fold variation
    accs = [m['accuracy'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    
    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)
    
    print(f"\n{'=' * 70}")
    print(f"  OVERALL RESULTS ({n} scenarios, 5-fold CV)")
    print(f"{'=' * 70}")
    print(f"  Accuracy:  {accuracy:.3f} ({acc_mean:.3f} ± {acc_std:.3f})")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f} ({f1_mean:.3f} ± {f1_std:.3f})")
    print(f"  TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")
    
    results = {
        'total_scenarios': int(n),
        'failure_scenarios': int(sum(y)),
        'stable_scenarios': int(n - sum(y)),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy_cv': f"{acc_mean:.3f} ± {acc_std:.3f}",
        'f1_cv': f"{f1_mean:.3f} ± {f1_std:.3f}",
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n_architectures': len(ARCHITECTURES),
        'n_failure_types': len(FAILURES),
        'n_seeds': len(seeds),
        'fold_metrics': fold_metrics,
    }
    
    with open('experiments/prediction_200_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/prediction_200_results.json")

if __name__ == '__main__':
    main()
