"""
Overhead Measurement Experiment
Measures per-component timing for ARC's monitoring subsystems.
Backs Table 9 (Computational Overhead) in the paper.

Measures:
  - Gradient norm computation
  - Weight statistics (norm + NaN check)
  - Loss analysis (threshold comparison + trend)
  - Checkpoint decision (deepcopy cost)
  - Forecasting (optimizer state check)
  - Total ARC overhead per step
  - Baseline forward+backward time (for relative overhead)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import copy
import math
import json
import numpy as np

DEVICE = "cpu"

def set_seed(s=42):
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

# ─── Models at different scales ────────────────────────────────────────────
class SmallMLP(nn.Module):
    """~500 params"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3*32*32, 16), nn.ReLU(), nn.Linear(16, 10)
        )
    def forward(self, x):
        return self.net(x)

class MediumCNN(nn.Module):
    """~340K params"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*4*4, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class LargeCNN(nn.Module):
    """~2.5M params"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

MODELS = {
    'SmallMLP (~500)': SmallMLP,
    'MediumCNN (~340K)': MediumCNN,
    'LargeCNN (~2.5M)': LargeCNN,
}

def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─── Individual Component Timing ──────────────────────────────────────────
def time_gradient_norm(model, n_iters=100):
    """Time: compute gradient norm across all parameters."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        _ = math.sqrt(total)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def time_weight_statistics(model, n_iters=100):
    """Time: weight norm + NaN/Inf check."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        total_wnorm = 0.0
        healthy = True
        for p in model.parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                healthy = False
            total_wnorm += p.data.norm(2).item() ** 2
        _ = math.sqrt(total_wnorm)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def time_loss_analysis(losses, n_iters=100):
    """Time: loss threshold check + trend computation."""
    times = []
    baseline = np.mean(losses[-5:]) if len(losses) >= 5 else 2.5
    threshold = max(baseline * 3.0, 10.0)
    for _ in range(n_iters):
        t0 = time.perf_counter()
        curr_loss = losses[-1] if losses else 2.5
        anomaly = curr_loss > threshold
        if len(losses) >= 2:
            trend = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)
            var = np.var(losses[-10:])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def time_checkpoint_decision(model, optimizer, n_iters=20):
    """Time: deepcopy model + optimizer state."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        checkpoint = copy.deepcopy(model.state_dict())
        opt_checkpoint = copy.deepcopy(optimizer.state_dict())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def time_forecasting(optimizer, n_iters=100):
    """Time: optimizer state norm check (forecasting proxy)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        total_norm = 0.0
        count = 0
        for state in optimizer.state.values():
            if 'momentum_buffer' in state:
                total_norm += state['momentum_buffer'].norm(2).item() ** 2
                count += 1
        if count > 0:
            _ = math.sqrt(total_norm)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)

def time_forward_backward(model, criterion, x, y, optimizer, n_iters=50):
    """Time: baseline forward + backward + step."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times)


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  OVERHEAD MEASUREMENT EXPERIMENT")
    print("=" * 70)
    
    loader = get_cifar10_loader(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    
    all_results = {}
    
    for model_name, model_fn in MODELS.items():
        set_seed(42)
        model = model_fn().to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        loader_iter = cycle_loader(loader)
        params = count_params(model)
        
        print(f"\n{'=' * 70}")
        print(f"  Model: {model_name} ({params:,} params)")
        print(f"{'=' * 70}")
        
        # Warmup: run a few steps to populate optimizer state and gradients
        losses = []
        for step in range(10):
            x, y = next(loader_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # Get a batch for timing
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Do one more forward/backward to ensure grads exist
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Time each component
        t_grad = time_gradient_norm(model)
        t_weight = time_weight_statistics(model)
        t_loss = time_loss_analysis(losses)
        t_checkpoint = time_checkpoint_decision(model, optimizer)
        t_forecast = time_forecasting(optimizer)
        t_baseline = time_forward_backward(model, criterion, x, y, optimizer)
        
        t_total_arc = t_grad + t_weight + t_loss + t_forecast
        # Checkpoint is amortized (every 10 steps), so per-step cost is /10
        t_checkpoint_amortized = t_checkpoint / 10.0
        t_total_with_ckpt = t_total_arc + t_checkpoint_amortized
        
        relative_overhead = (t_total_with_ckpt / t_baseline) * 100
        
        print(f"\n  Component Timing (median of 100 iterations):")
        print(f"  {'Gradient Norm':<25s} {t_grad:>8.3f} ms  ({t_grad/t_total_with_ckpt*100:>5.1f}%)")
        print(f"  {'Weight Statistics':<25s} {t_weight:>8.3f} ms  ({t_weight/t_total_with_ckpt*100:>5.1f}%)")
        print(f"  {'Loss Analysis':<25s} {t_loss:>8.3f} ms  ({t_loss/t_total_with_ckpt*100:>5.1f}%)")
        print(f"  {'Checkpoint (amortized)':<25s} {t_checkpoint_amortized:>8.3f} ms  ({t_checkpoint_amortized/t_total_with_ckpt*100:>5.1f}%)")
        print(f"  {'Forecasting':<25s} {t_forecast:>8.3f} ms  ({t_forecast/t_total_with_ckpt*100:>5.1f}%)")
        print(f"  {'─'*50}")
        print(f"  {'Total ARC Overhead':<25s} {t_total_with_ckpt:>8.3f} ms")
        print(f"  {'Baseline (fwd+bwd+step)':<25s} {t_baseline:>8.3f} ms")
        print(f"  {'Relative Overhead':<25s} {relative_overhead:>7.1f}%")
        
        all_results[model_name] = {
            'params': params,
            'gradient_norm_ms': round(t_grad, 3),
            'weight_stats_ms': round(t_weight, 3),
            'loss_analysis_ms': round(t_loss, 3),
            'checkpoint_amortized_ms': round(t_checkpoint_amortized, 3),
            'checkpoint_full_ms': round(t_checkpoint, 3),
            'forecasting_ms': round(t_forecast, 3),
            'total_arc_overhead_ms': round(t_total_with_ckpt, 3),
            'baseline_fwd_bwd_ms': round(t_baseline, 3),
            'relative_overhead_pct': round(relative_overhead, 1),
        }
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"  OVERHEAD SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25s} {'Params':>10s} {'ARC (ms)':>10s} {'Base (ms)':>10s} {'Overhead':>10s}")
    print(f"  {'-'*65}")
    for name, r in all_results.items():
        print(f"  {name:<25s} {r['params']:>10,} {r['total_arc_overhead_ms']:>9.3f} {r['baseline_fwd_bwd_ms']:>9.3f} {r['relative_overhead_pct']:>9.1f}%")
    
    with open('experiments/overhead_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to experiments/overhead_results.json")

if __name__ == '__main__':
    main()
