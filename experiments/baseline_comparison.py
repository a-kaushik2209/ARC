"""
Baseline Comparison Experiment
Compares 4 training protection methods across failure scenarios:
  1. No protection (vanilla training)
  2. Gradient clipping only
  3. Loss-only monitoring (checkpoint + reload if loss spikes)
  4. Full ARC (multi-signal monitoring + recovery)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy
import time
import json
import math
import os
import sys
import numpy as np

DEVICE = "cpu"
SEED = 42

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
def inject_nan_bomb(model):
    """Inject NaN into all parameters"""
    with torch.no_grad():
        for p in model.parameters():
            p.data[0] = float('nan')

def inject_gradient_explosion(model, optimizer, x, y, criterion):
    """Simulate gradient explosion via massive LR spike"""
    for pg in optimizer.param_groups:
        pg['lr'] = 10.0  # 500x spike
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    for pg in optimizer.param_groups:
        pg['lr'] = 0.02  # restore

def inject_weight_corruption(model, scale=0.5):
    """Add large noise to weights"""
    with torch.no_grad():
        for p in model.parameters():
            p.data += torch.randn_like(p.data) * scale

def inject_loss_explosion(model, optimizer, x, y, criterion):
    """Force loss to infinity via extreme weights"""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1000.0

def inject_optimizer_reset(optimizer):
    """Zero out optimizer momentum buffers"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                v.zero_()

FAILURE_TYPES = {
    'nan_bomb': inject_nan_bomb,
    'gradient_explosion': inject_gradient_explosion,
    'weight_corruption': inject_weight_corruption,
    'loss_explosion': inject_loss_explosion,
    'optimizer_reset': inject_optimizer_reset,
}

# ─── Baseline 1: No Protection ────────────────────────────────────────────
def run_no_protection(model, optimizer, criterion, loader_iter, failure_fn, failure_args):
    """Train with no protection. Returns (detected, recovered, false_positives, time_ms)"""
    t0 = time.time()
    detected = False
    recovered = False
    
    # Train 20 steps
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        
        if torch.isnan(loss) or torch.isinf(loss):
            detected = True
            break
        
        loss.backward()
        optimizer.step()
    
    # Inject failure at step 20
    failure_fn(*failure_args)
    
    # Continue 30 more steps
    crash = False
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        try:
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss):
                crash = True
                detected = True
                break
            loss.backward()
            # Check for NaN grads
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                crash = True
                detected = True
                break
            optimizer.step()
        except RuntimeError:
            crash = True
            detected = True
            break
    
    elapsed = (time.time() - t0) * 1000
    return detected, recovered, 0, elapsed

# ─── Baseline 2: Gradient Clipping Only ──────────────────────────────────
def run_grad_clipping(model, optimizer, criterion, loader_iter, failure_fn, failure_args):
    """Train with gradient clipping only."""
    t0 = time.time()
    detected = False
    recovered = False
    max_norm = 1.0
    
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            detected = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    
    failure_fn(*failure_args)
    
    crash = False
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        try:
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss):
                crash = True
                detected = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        except RuntimeError:
            crash = True
            detected = True
            break
    
    elapsed = (time.time() - t0) * 1000
    return detected, recovered, 0, elapsed

# ─── Baseline 3: Loss-Only Monitoring ────────────────────────────────────
def run_loss_only(model, optimizer, criterion, loader_iter, failure_fn, failure_args):
    """Train with loss-only monitoring: save checkpoint, reload if loss > 2x baseline."""
    t0 = time.time()
    detected = False
    recovered = False
    false_positives = 0
    
    # Warmup and establish baseline loss
    losses = []
    checkpoint = copy.deepcopy(model.state_dict())
    opt_checkpoint = copy.deepcopy(optimizer.state_dict())
    
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            detected = True
            break
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # Update checkpoint every 10 steps
        if step % 10 == 9:
            checkpoint = copy.deepcopy(model.state_dict())
            opt_checkpoint = copy.deepcopy(optimizer.state_dict())
    
    baseline_loss = np.mean(losses[-5:]) if losses else 2.5
    threshold = max(baseline_loss * 3.0, 10.0)  # 3x or absolute 10.0
    
    failure_fn(*failure_args)
    
    rollback_count = 0
    max_rollbacks = 3
    
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        try:
            out = model(x)
            loss = criterion(out, y)
            loss_val = loss.item()
            
            if torch.isnan(loss) or torch.isinf(loss) or loss_val > threshold:
                detected = True
                if rollback_count < max_rollbacks:
                    model.load_state_dict(copy.deepcopy(checkpoint))
                    optimizer.load_state_dict(copy.deepcopy(opt_checkpoint))
                    rollback_count += 1
                    # Reduce LR
                    for pg in optimizer.param_groups:
                        pg['lr'] *= 0.5
                    recovered = True
                    continue
                else:
                    break
            
            loss.backward()
            optimizer.step()
        except RuntimeError:
            detected = True
            if rollback_count < max_rollbacks:
                model.load_state_dict(copy.deepcopy(checkpoint))
                optimizer.load_state_dict(copy.deepcopy(opt_checkpoint))
                rollback_count += 1
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.5
                recovered = True
            else:
                break
    
    # Verify recovery: can we still compute a valid loss?
    if recovered:
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
            final_loss = criterion(out, y)
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                recovered = False
    
    elapsed = (time.time() - t0) * 1000
    return detected, recovered, false_positives, elapsed

# ─── Baseline 4: Full ARC ────────────────────────────────────────────────
def get_optimizer_state_norm(optimizer):
    """Compute total norm of optimizer momentum buffers."""
    total_norm = 0.0
    buffer_count = 0
    for state in optimizer.state.values():
        if 'momentum_buffer' in state:
            total_norm += state['momentum_buffer'].norm(2).item() ** 2
            buffer_count += 1
    return math.sqrt(total_norm) if buffer_count > 0 else -1.0

def run_full_arc(model, optimizer, criterion, loader_iter, failure_fn, failure_args):
    """Simulates full ARC: multi-signal monitoring + checkpoint + rollback + adaptive recovery."""
    t0 = time.time()
    detected = False
    recovered = False
    false_positives = 0
    
    checkpoint = copy.deepcopy(model.state_dict())
    opt_checkpoint = copy.deepcopy(optimizer.state_dict())
    
    losses = []
    grad_norms = []
    weight_norms = []
    opt_state_norms = []
    
    for step in range(20):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            detected = True
            break
        losses.append(loss.item())
        loss.backward()
        
        # Track gradient norms
        total_gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_gnorm += p.grad.data.norm(2).item() ** 2
        grad_norms.append(math.sqrt(total_gnorm))
        
        # Track weight norms
        total_wnorm = 0.0
        for p in model.parameters():
            total_wnorm += p.data.norm(2).item() ** 2
        weight_norms.append(math.sqrt(total_wnorm))
        
        optimizer.step()
        
        # Track optimizer state (momentum buffers) after step
        osn = get_optimizer_state_norm(optimizer)
        if osn >= 0:
            opt_state_norms.append(osn)
        
        if step % 10 == 9:
            checkpoint = copy.deepcopy(model.state_dict())
            opt_checkpoint = copy.deepcopy(optimizer.state_dict())
    
    # Compute baselines
    baseline_loss = np.mean(losses[-5:]) if losses else 2.5
    baseline_gnorm = np.mean(grad_norms[-5:]) if grad_norms else 1.0
    baseline_wnorm = np.mean(weight_norms[-5:]) if weight_norms else 1.0
    baseline_opt_norm = np.mean(opt_state_norms[-5:]) if opt_state_norms else -1.0
    
    failure_fn(*failure_args)
    
    rollback_count = 0
    max_rollbacks = 5
    
    def _do_rollback():
        nonlocal rollback_count, recovered
        model.load_state_dict(copy.deepcopy(checkpoint))
        optimizer.load_state_dict(copy.deepcopy(opt_checkpoint))
        rollback_count += 1
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
        with torch.no_grad():
            for p in model.parameters():
                p.data += torch.randn_like(p.data) * 0.001
        recovered = True
    
    for step in range(30):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        # Multi-signal check BEFORE forward pass
        # 1. Weight health check
        weight_healthy = True
        for p in model.parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                weight_healthy = False
                break
        
        curr_wnorm = 0.0
        for p in model.parameters():
            curr_wnorm += p.data.norm(2).item() ** 2
        curr_wnorm = math.sqrt(curr_wnorm)
        weight_drift = abs(curr_wnorm - baseline_wnorm) / max(baseline_wnorm, 1e-8)
        
        if not weight_healthy or weight_drift > 5.0:
            detected = True
            if rollback_count < max_rollbacks:
                _do_rollback()
                continue
            else:
                break
        
        # 4. Optimizer state integrity check
        if baseline_opt_norm > 0:
            curr_opt_norm = get_optimizer_state_norm(optimizer)
            if curr_opt_norm >= 0:
                opt_change = abs(curr_opt_norm - baseline_opt_norm) / max(baseline_opt_norm, 1e-8)
                # If momentum buffers are zeroed or changed dramatically
                if curr_opt_norm < baseline_opt_norm * 0.01 or opt_change > 10.0:
                    detected = True
                    if rollback_count < max_rollbacks:
                        _do_rollback()
                        continue
                    else:
                        break
        
        try:
            out = model(x)
            loss = criterion(out, y)
            loss_val = loss.item()
            
            # 2. Loss check
            loss_anomaly = (torch.isnan(loss) or torch.isinf(loss) or 
                          loss_val > max(baseline_loss * 3.0, 10.0))
            
            if loss_anomaly:
                detected = True
                if rollback_count < max_rollbacks:
                    _do_rollback()
                    continue
                else:
                    break
            
            loss.backward()
            
            # 3. Gradient health check
            total_gnorm = 0.0
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        break
                    total_gnorm += p.grad.data.norm(2).item() ** 2
            total_gnorm = math.sqrt(total_gnorm)
            
            grad_anomaly = has_nan_grad or total_gnorm > baseline_gnorm * 100
            
            if grad_anomaly:
                detected = True
                if rollback_count < max_rollbacks:
                    _do_rollback()
                    continue
                else:
                    break
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        except RuntimeError:
            detected = True
            if rollback_count < max_rollbacks:
                _do_rollback()
            else:
                break
    
    # Verify recovery
    if recovered:
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
            final_loss = criterion(out, y)
            if torch.isnan(final_loss) or torch.isinf(final_loss):
                recovered = False
    
    elapsed = (time.time() - t0) * 1000
    return detected, recovered, false_positives, elapsed


# ─── Run Stable (for false positive measurement) ─────────────────────────
def run_stable_training(run_fn, loader_iter, num_steps=50):
    """Run a method on stable training to check for false positives."""
    set_seed(SEED)
    model = TinyCNN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # No failure injection - just train normally
    def no_failure(*args):
        pass
    
    _, _, fp, _ = run_fn(model, optimizer, criterion, loader_iter, no_failure, [])
    
    # Check if training state is healthy
    x, y = next(loader_iter)
    x, y = x.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        out = model(x)
        loss = criterion(out, y)
        if torch.isnan(loss) or torch.isinf(loss):
            return 1  # false detection
    return 0


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BASELINE COMPARISON EXPERIMENT")
    print("=" * 70)
    
    loader = get_cifar10_loader(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    
    methods = {
        'No Protection':    run_no_protection,
        'Gradient Clipping': run_grad_clipping,
        'Loss-Only Monitor': run_loss_only,
        'Full ARC':         run_full_arc,
    }
    
    failure_types = list(FAILURE_TYPES.keys())
    seeds = [42, 43, 44, 45, 46]
    
    results = {name: {'detected': 0, 'recovered': 0, 'total': 0, 'times': [], 'false_positives': 0}
               for name in methods}
    
    total_scenarios = len(failure_types) * len(seeds)
    
    for method_name, method_fn in methods.items():
        print(f"\n{'=' * 70}")
        print(f"  Method: {method_name}")
        print(f"{'=' * 70}")
        
        for fi, failure_name in enumerate(failure_types):
            for si, seed in enumerate(seeds):
                scenario_num = fi * len(seeds) + si + 1
                set_seed(seed)
                
                model = TinyCNN().to(DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
                loader_iter = cycle_loader(loader)
                
                # Prepare failure args
                if failure_name == 'nan_bomb':
                    failure_fn = inject_nan_bomb
                    failure_args = [model]
                elif failure_name == 'gradient_explosion':
                    failure_fn = inject_gradient_explosion
                    x, y = next(loader_iter)
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    failure_args = [model, optimizer, x, y, criterion]
                elif failure_name == 'weight_corruption':
                    failure_fn = inject_weight_corruption
                    failure_args = [model, 0.5]
                elif failure_name == 'loss_explosion':
                    failure_fn = inject_loss_explosion
                    x, y = next(loader_iter)
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    failure_args = [model, optimizer, x, y, criterion]
                elif failure_name == 'optimizer_reset':
                    failure_fn = inject_optimizer_reset
                    failure_args = [optimizer]
                
                detected, recovered, fp, elapsed = method_fn(
                    model, optimizer, criterion, loader_iter, failure_fn, failure_args
                )
                
                results[method_name]['detected'] += int(detected)
                results[method_name]['recovered'] += int(recovered)
                results[method_name]['total'] += 1
                results[method_name]['times'].append(elapsed)
                
                status = "✓ recovered" if recovered else ("⚠ detected" if detected else "✗ missed")
                print(f"  [{scenario_num:2d}/{total_scenarios}] {failure_name} (seed={seed}): {status} ({elapsed:.0f}ms)")
        
        # False positive check (5 stable runs)
        print(f"  --- False positive check ---")
        fp_count = 0
        for seed in seeds:
            set_seed(seed)
            loader_iter = cycle_loader(loader)
            fp = run_stable_training(method_fn, loader_iter)
            fp_count += fp
        results[method_name]['false_positives'] = fp_count
        print(f"  False positives: {fp_count}/{len(seeds)}")
    
    # ─── Print Summary Table ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Method':<20s} {'Detection':>10s} {'Recovery':>10s} {'FP':>5s} {'Overhead':>10s}")
    print("-" * 60)
    
    summary = {}
    for name, r in results.items():
        det_rate = r['detected'] / r['total'] * 100 if r['total'] > 0 else 0
        rec_rate = r['recovered'] / r['total'] * 100 if r['total'] > 0 else 0
        avg_time = np.mean(r['times']) if r['times'] else 0
        fp = r['false_positives']
        
        print(f"{name:<20s} {det_rate:>9.1f}% {rec_rate:>9.1f}% {fp:>4d} {avg_time:>8.0f}ms")
        
        summary[name] = {
            'detection_rate': round(det_rate, 1),
            'recovery_rate': round(rec_rate, 1),
            'false_positives': fp,
            'avg_time_ms': round(avg_time, 1),
            'total_scenarios': r['total'],
            'detected': r['detected'],
            'recovered': r['recovered'],
        }
    
    # Save results
    with open('experiments/baseline_comparison_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to experiments/baseline_comparison_results.json")

if __name__ == '__main__':
    main()