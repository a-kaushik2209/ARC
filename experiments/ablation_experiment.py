"""
Ablation Experiment
Backs Table 5 of the paper by systematically disabling each ARC monitoring
component and measuring detection rate across failure scenarios.

Components tested:
  - Full ARC (all components)
  - No weight health monitoring
  - No gradient monitoring
  - No loss monitoring
  - No forecasting (optimizer state check)
  - Loss-only baseline
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import math
import json
import numpy as np

DEVICE = "cpu"

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

# ─── Failure Injectors ─────────────────────────────────────────────────────
def inject_nan_bomb(model, optimizer):
    with torch.no_grad():
        for p in model.parameters():
            p.data[0] = float('nan')

def inject_gradient_explosion(model, optimizer):
    for pg in optimizer.param_groups:
        pg['lr'] = 10.0

def inject_weight_corruption(model, optimizer):
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * 0.5

def inject_loss_explosion(model, optimizer):
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1000.0

def inject_optimizer_reset(model, optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                v.zero_()

def inject_silent_corruption(model, optimizer):
    """Corrupt 10% of weights slightly - hard to detect"""
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p.data) < 0.1
            p.data[mask] += torch.randn_like(p.data[mask]) * 0.05

def inject_lr_spike(model, optimizer):
    for pg in optimizer.param_groups:
        pg['lr'] = 5.0

FAILURES = {
    'nan_bomb': inject_nan_bomb,
    'gradient_explosion': inject_gradient_explosion,
    'weight_corruption': inject_weight_corruption,
    'loss_explosion': inject_loss_explosion,
    'optimizer_reset': inject_optimizer_reset,
    'silent_corruption': inject_silent_corruption,
    'lr_spike': inject_lr_spike,
}

# ─── Helper Analytics Metrics ──────────────────────────────────────────────
def get_optimizer_state_norm(optimizer):
    total_norm = 0.0
    count = 0
    for state in optimizer.state.values():
        if 'momentum_buffer' in state:
            total_norm += state['momentum_buffer'].norm(2).item() ** 2
            count += 1
    return math.sqrt(total_norm) if count > 0 else -1.0

def calculate_weight_norm(model):
    wnorm = 0.0
    for p in model.parameters():
        wnorm += p.data.norm(2).item() ** 2
    return math.sqrt(wnorm)

def calculate_gradient_norm(model):
    gnorm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gnorm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(gnorm)


# ─── Refactored Isolated Monitoring Sub-routines ─────────────────────────────
def check_weight_health(model, bl_wnorm):
    """Validates structural weight parameters for NaNs, Infs, or sudden drift."""
    for p in model.parameters():
        if torch.isnan(p.data).any() or torch.isinf(p.data).any():
            return True
            
    curr_wnorm = calculate_weight_norm(model)
    if abs(curr_wnorm - bl_wnorm) / max(bl_wnorm, 1e-8) > 5.0:
        return True
    return False

def check_optimizer_forecasting(optimizer, bl_opt):
    """Validates optimizer tracking states to catch structural anomalies."""
    if bl_opt <= 0:
        return False
        
    curr_opt = get_optimizer_state_norm(optimizer)
    if curr_opt >= 0:
        if curr_opt < bl_opt * 0.01 or abs(curr_opt - bl_opt) / max(bl_opt, 1e-8) > 10.0:
            return True
    return False

def check_loss_anomaly(loss, loss_val, bl_loss):
    """Evaluates whether the loss value has exploded or turned into a NaN context."""
    if torch.isnan(loss) or torch.isinf(loss):
        return True
    if loss_val > max(bl_loss * 3.0, 10.0):
        return True
    return False

def check_gradient_anomaly(model, bl_gnorm):
    """Monitors incoming backpropagation gradients for numerical instability."""
    for p in model.parameters():
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            return True
            
    gnorm = calculate_gradient_norm(model)
    if gnorm > bl_gnorm * 100:
        return True
    return False


# ─── Configurable ARC Detection ───────────────────────────────────────────
def run_arc_detection(model, optimizer, criterion, loader_iter, failure_fn,
                      use_weight_health=True,
                      use_gradient_monitor=True,
                      use_loss_monitor=True,
                      use_forecasting=True):
    """
    Run training with configurable ARC components.
    Returns True if failure was DETECTED.
    """
    losses, grad_norms, weight_norms, opt_norms = [], [], [], []
    
    # Warmup Loop Phase (20 steps)
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return True  # Detected anomaly immediately during warmup
            
        losses.append(loss.item())
        loss.backward()
        
        grad_norms.append(calculate_gradient_norm(model))
        weight_norms.append(calculate_weight_norm(model))
        
        optimizer.step()
        
        osn = get_optimizer_state_norm(optimizer)
        if osn >= 0:
            opt_norms.append(osn)
    
    # Establish Baselining Metrics
    bl_loss = np.mean(losses[-5:]) if losses else 2.5
    bl_gnorm = np.mean(grad_norms[-5:]) if grad_norms else 1.0
    bl_wnorm = np.mean(weight_norms[-5:]) if weight_norms else 1.0
    bl_opt = np.mean(opt_norms[-5:]) if opt_norms else -1.0
    
    # Inject targeting fault anomaly
    failure_fn(model, optimizer)
    
    # Evaluation Monitoring Loop Phase (30 steps)
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        # 1. Component Evaluation: Weight Health Validation
        if use_weight_health and check_weight_health(model, bl_wnorm):
            return True
        
        # 2. Component Evaluation: Optimizer Velocity Tracking
        if use_forecasting and check_optimizer_forecasting(optimizer, bl_opt):
            return True
        
        try:
            out = model(x)
            loss = criterion(out, y)
            loss_val = loss.item()
            
            # 3. Component Evaluation: Target Loss Verification
            if use_loss_monitor and check_loss_anomaly(loss, loss_val, bl_loss):
                return True
            
            loss.backward()
            
            # 4. Component Evaluation: Backpropagation Gradient Verification
            if use_gradient_monitor and check_gradient_anomaly(model, bl_gnorm):
                return True
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        except RuntimeError:
            return True
            
    return False  # Failure was not caught by any enabled active tracker


# ─── Loss-Only Baseline ──────────────────────────────────────────────────
def run_loss_only(model, optimizer, criterion, loader_iter, failure_fn):
    """Only monitor loss — no weight/gradient/optimizer checks."""
    losses = []
    
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            return True
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    bl_loss = np.mean(losses[-5:]) if losses else 2.5
    threshold = max(bl_loss * 3.0, 10.0)
    
    failure_fn(model, optimizer)
    
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        try:
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > threshold:
                return True
            loss.backward()
            optimizer.step()
        except RuntimeError:
            return True
    
    return False


# ─── Main Execution Workflow ───────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ABLATION EXPERIMENT — Component Detection Impact")
    print("=" * 70)
    
    loader = get_cifar10_loader(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    seeds = [42, 43, 44, 45, 46]
    failure_names = list(FAILURES.keys())
    total_scenarios = len(failure_names) * len(seeds)
    
    configs = {
        'Full ARC (all)': dict(use_weight_health=True, use_gradient_monitor=True,
                                use_loss_monitor=True, use_forecasting=True),
        '- Weight Health':  dict(use_weight_health=False, use_gradient_monitor=True,
                                use_loss_monitor=True, use_forecasting=True),
        '- Gradient Mon.':  dict(use_weight_health=True, use_gradient_monitor=False,
                                use_loss_monitor=True, use_forecasting=True),
        '- Loss Monitor':   dict(use_weight_health=True, use_gradient_monitor=True,
                                use_loss_monitor=False, use_forecasting=True),
        '- Forecasting':    dict(use_weight_health=True, use_gradient_monitor=True,
                                use_loss_monitor=True, use_forecasting=False),
    }
    
    results = {}
    
    for config_name, kwargs in configs.items():
        detected = 0
        total = 0
        
        print(f"\n  Config: {config_name}")
        for fi, fname in enumerate(failure_names):
            for seed in seeds:
                set_seed(seed)
                model = TinyCNN().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                d = run_arc_detection(model, optimizer, criterion, loader_iter,
                                      FAILURES[fname], **kwargs)
                detected += int(d)
                total += 1
        
        rate = detected / total * 100
        results[config_name] = {'detected': detected, 'total': total, 'rate': round(rate, 1)}
        print(f"    Detection: {detected}/{total} = {rate:.1f}%")
    
    # Loss-only baseline
    print(f"\n  Config: Loss Only (baseline)")
    detected = 0
    total = 0
    for fname in failure_names:
        for seed in seeds:
            set_seed(seed)
            model = TinyCNN().to(DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            loader_iter = cycle_loader(loader)
            
            d = run_loss_only(model, optimizer, criterion, loader_iter, FAILURES[fname])
            detected += int(d)
            total += 1
    
    rate = detected / total * 100
    results['Loss Only (baseline)'] = {'detected': detected, 'total': total, 'rate': round(rate, 1)}
    print(f"    Detection: {detected}/{total} = {rate:.1f}%")
    
    # Summary Table Construction
    print(f"\n{'=' * 70}")
    print(f"  ABLATION RESULTS ({total_scenarios} scenarios each)")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<30s} {'Detection':>10s} {'Δ from Full':>12s}")
    print(f"  {'-'*55}")
    
    full_rate = results['Full ARC (all)']['rate']
    for name, r in results.items():
        delta = r['rate'] - full_rate
        delta_str = f"{delta:+.1f}%" if name != 'Full ARC (all)' else '---'
        delta_str = '---' if name == 'Full ARC (all)' else f"{delta:+.1f}%"
        print(f"  {name:<30s} {r['rate']:>9.1f}% {delta_str:>12s}")
    
    with open('experiments/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/ablation_results.json")

if __name__ == '__main__':
    main()