"""
Improved Prediction Experiment (200+ scenarios, v2)
Improvements over v1:
  - 12 features instead of 8 (added optimizer state norm, loss 2nd derivative, 
    gradient entropy, weight norm acceleration)
  - 2-layer MLP classifier instead of logistic regression
  - Same 200 scenarios for fair comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import math
import os

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
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 100.0

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

# ─── Enhanced Feature Extraction (12 features) ────────────────────────────
def get_optimizer_state_norm(optimizer):
    total_norm = 0.0
    count = 0
    for state in optimizer.state.values():
        if 'momentum_buffer' in state:
            total_norm += state['momentum_buffer'].norm(2).item() ** 2
            count += 1
    return math.sqrt(total_norm) if count > 0 else 0.0

def compute_gradient_entropy(model):
    """Entropy of gradient magnitude distribution across parameters."""
    grad_mags = []
    for p in model.parameters():
        if p.grad is not None:
            grad_mags.append(p.grad.data.abs().mean().item())
    if not grad_mags or sum(grad_mags) == 0:
        return 0.0
    mags = np.array(grad_mags)
    mags = mags / (mags.sum() + 1e-10)
    entropy = -np.sum(mags * np.log(mags + 1e-10))
    return entropy

def collect_enhanced_features(model, optimizer, criterion, loader_iter, num_steps=30,
                               inject_failure=False, failure_fn=None, inject_at=15):
    """
    12 features:
    0: loss_trend - relative change from first to last loss
    1: loss_var - variance of losses
    2: grad_mean - mean gradient norm
    3: grad_var - variance of gradient norms
    4: grad_max - max gradient norm
    5: weight_norm_change - relative weight norm change
    6: weight_var - variance of weight norms
    7: nan_count - count of NaN/Inf events
    8: opt_state_change - relative change in optimizer momentum norm
    9: loss_acceleration - second derivative of loss (how fast loss changes)
    10: grad_entropy_change - change in gradient entropy
    11: weight_norm_accel - acceleration of weight norm changes
    """
    losses = []
    grad_norms = []
    weight_norms = []
    nan_count = 0
    grad_entropies = []
    
    initial_wnorm = 0.0
    for p in model.parameters():
        initial_wnorm += p.data.norm(2).item() ** 2
    initial_wnorm = math.sqrt(initial_wnorm)
    
    initial_opt_norm = get_optimizer_state_norm(optimizer)
    
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
                losses.append(100.0)
                grad_norms.append(100.0)
                grad_entropies.append(0.0)
                curr_wnorm = 0.0
                for p in model.parameters():
                    curr_wnorm += p.data.norm(2).item() ** 2
                weight_norms.append(math.sqrt(curr_wnorm))
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
            
            # Gradient entropy
            ge = compute_gradient_entropy(model)
            grad_entropies.append(ge)
            
            if not has_nan:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        
        except RuntimeError:
            nan_count += 1
            losses.append(100.0)
            grad_norms.append(100.0)
            grad_entropies.append(0.0)
        
        curr_wnorm = 0.0
        for p in model.parameters():
            curr_wnorm += p.data.norm(2).item() ** 2
        weight_norms.append(math.sqrt(curr_wnorm))
    
    # Compute features
    if len(losses) < 3:
        return [0.0] * 12
    
    losses = np.array(losses)
    grad_norms = np.array(grad_norms)
    weight_norms = np.array(weight_norms)
    grad_entropies = np.array(grad_entropies) if grad_entropies else np.array([0.0])
    
    # Original 8 features
    loss_trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)
    loss_var = np.var(losses)
    grad_mean = np.mean(grad_norms)
    grad_var = np.var(grad_norms)
    grad_max = np.max(grad_norms)
    weight_norm_change = (weight_norms[-1] - initial_wnorm) / max(abs(initial_wnorm), 1e-8)
    weight_var = np.var(weight_norms)
    
    # NEW: 4 additional features
    # 8. Optimizer state change
    final_opt_norm = get_optimizer_state_norm(optimizer)
    opt_state_change = abs(final_opt_norm - initial_opt_norm) / max(abs(initial_opt_norm), 1e-8)
    
    # 9. Loss acceleration (2nd derivative)
    if len(losses) >= 3:
        loss_diffs = np.diff(losses)
        loss_accel = np.mean(np.abs(np.diff(loss_diffs))) if len(loss_diffs) >= 2 else 0.0
    else:
        loss_accel = 0.0
    
    # 10. Gradient entropy change
    if len(grad_entropies) >= 2:
        ge_change = abs(grad_entropies[-1] - grad_entropies[0])
    else:
        ge_change = 0.0
    
    # 11. Weight norm acceleration
    if len(weight_norms) >= 3:
        wn_diffs = np.diff(weight_norms)
        wn_accel = np.mean(np.abs(np.diff(wn_diffs))) if len(wn_diffs) >= 2 else 0.0
    else:
        wn_accel = 0.0
    
    features = [loss_trend, loss_var, grad_mean, grad_var, grad_max,
                weight_norm_change, weight_var, float(nan_count),
                opt_state_change, loss_accel, ge_change, wn_accel]
    
    # Sanitize
    features = [0.0 if (math.isnan(f) or math.isinf(f)) else f for f in features]
    
    return features


# ─── MLP Classifier ──────────────────────────────────────────────────────
class MLPClassifier:
    """Simple 2-layer MLP for binary classification."""
    def __init__(self, n_features, hidden=32):
        self.W1 = np.random.randn(n_features, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * 0.1
        self.b2 = np.zeros(1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def forward(self, X):
        self.h = self.relu(X @ self.W1 + self.b1)
        self.out = self.sigmoid(self.h @ self.W2 + self.b2).flatten()
        return self.out
    
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
    
    def fit(self, X, y, lr=0.01, epochs=2000):
        n = len(y)
        for epoch in range(epochs):
            # Forward
            p = self.forward(X)
            
            # Backward
            error = (p - y).reshape(-1, 1)  # (n, 1)
            
            # Output layer
            dW2 = self.h.T @ error / n
            db2 = np.mean(error, axis=0)
            
            # Hidden layer
            dh = error @ self.W2.T  # (n, hidden)
            dh[self.h <= 0] = 0  # ReLU derivative
            
            dW1 = X.T @ dh / n
            db1 = np.mean(dh, axis=0)
            
            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2


# ─── Logistic Regression (for comparison) ────────────────────────────────
class LogisticRegression:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)
    
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
    print("  IMPROVED PREDICTION EXPERIMENT (200 Scenarios, v2)")
    print("=" * 70)
    
    loader = get_cifar10_loader(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    seeds = [42, 43, 44, 45, 46]
    
    all_features = []
    all_labels = []
    count = 0
    total = len(ARCHITECTURES) * len(FAILURES) * len(seeds) * 2
    
    for arch_name, arch_fn in ARCHITECTURES.items():
        for fail_name, fail_fn in FAILURES.items():
            for seed in seeds:
                # Failure scenario
                count += 1
                set_seed(seed)
                model = arch_fn().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                features = collect_enhanced_features(
                    model, optimizer, criterion, loader_iter,
                    num_steps=30, inject_failure=True, failure_fn=fail_fn, inject_at=15
                )
                all_features.append(features)
                all_labels.append(1)
                
                if count % 20 == 0 or count <= 4:
                    print(f"  [{count:3d}/{total}] {arch_name}/{fail_name} seed={seed} FAILURE")
                
                # Stable scenario
                count += 1
                set_seed(seed)
                model = arch_fn().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                features = collect_enhanced_features(
                    model, optimizer, criterion, loader_iter,
                    num_steps=30, inject_failure=False
                )
                all_features.append(features)
                all_labels.append(0)
                
                if count % 20 == 0 or count <= 4:
                    print(f"  [{count:3d}/{total}] {arch_name}/{fail_name} seed={seed} STABLE")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n  Total scenarios: {len(y)} ({sum(y)} failure, {len(y) - sum(y)} stable)")
    print(f"  Features per scenario: {X.shape[1]} (was 8 in v1, now 12)")
    
    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    
    # 5-fold CV for both classifiers
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // 5
    
    for clf_name, clf_class, clf_kwargs in [
        ("Logistic Regression (12 feat)", LogisticRegression, {'lr': 0.1, 'epochs': 2000}),
        ("MLP Classifier (12 feat)", MLPClassifier, {'lr': 0.05, 'epochs': 3000}),
    ]:
        print(f"\n{'=' * 70}")
        print(f"  Classifier: {clf_name}")
        print(f"{'=' * 70}")
        
        all_preds = np.zeros(n)
        all_true = np.zeros(n)
        fold_metrics = []
        
        for fold in range(5):
            test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            
            X_train, y_train = X_norm[train_idx], y[train_idx]
            X_test, y_test = X_norm[test_idx], y[test_idx]
            
            np.random.seed(fold + 100)
            if 'MLP' in clf_name:
                clf = MLPClassifier(X_train.shape[1], hidden=32)
            else:
                clf = LogisticRegression(X_train.shape[1])
            clf.fit(X_train, y_train, **clf_kwargs)
            
            preds = clf.predict(X_test)
            all_preds[test_idx] = preds
            all_true[test_idx] = y_test
            
            tp = np.sum((preds == 1) & (y_test == 1))
            fp = np.sum((preds == 1) & (y_test == 0))
            fn = np.sum((preds == 0) & (y_test == 1))
            tn = np.sum((preds == 0) & (y_test == 0))
            
            acc = (tp + tn) / len(y_test)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            
            fold_metrics.append({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
            print(f"  Fold {fold+1}: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}")
        
        # Overall
        tp = np.sum((all_preds == 1) & (all_true == 1))
        fp = np.sum((all_preds == 1) & (all_true == 0))
        fn = np.sum((all_preds == 0) & (all_true == 1))
        tn = np.sum((all_preds == 0) & (all_true == 0))
        
        accuracy = (tp + tn) / n
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        accs = [m['accuracy'] for m in fold_metrics]
        f1s = [m['f1'] for m in fold_metrics]
        
        print(f"\n  OVERALL ({clf_name}):")
        print(f"  Accuracy:  {accuracy:.3f} ({np.mean(accs):.3f} ± {np.std(accs):.3f})")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f} ({np.mean(f1s):.3f} ± {np.std(f1s):.3f})")
        print(f"  TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")
    
    # Save best results (MLP with 12 features)
    results = {
        'total_scenarios': int(n),
        'n_features': 12,
        'classifiers_tested': ['LogisticRegression_12feat', 'MLP_12feat'],
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy_cv': f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        'f1_cv': f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'improvements': 'Added optimizer state norm, loss acceleration, gradient entropy change, weight norm acceleration features. Used MLP classifier.',
    }
    
    with open('experiments/prediction_200_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/prediction_200_v2_results.json")

if __name__ == '__main__':
    main()
