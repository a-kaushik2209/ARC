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
ARC Overhead Analysis (Phase 5)

Addresses reviewer concern: "You listed time overhead but omitted memory overhead"

This module measures:
1. Time overhead (wall clock)
2. Memory overhead (peak RAM/VRAM)
3. Checkpoint size
4. Per-step latency

For models of different sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Models of Different Sizes
# =============================================================================

def create_model(size="small"):
    """Create model of specified size."""
    
    if size == "small":
        # ~100K params
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    elif size == "medium":
        # ~1M params
        return nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    elif size == "large":
        # ~10M params
        return nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
    
    elif size == "xlarge":
        # ~50M params
        return nn.Sequential(
            nn.Linear(100, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )


def get_model_size_bytes(model):
    """Get model size in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


# =============================================================================
# Overhead Measurement
# =============================================================================

def measure_memory():
    """Measure current memory usage."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    else:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss


def run_overhead_test(model_size, use_arc=True, n_steps=100):
    """Measure overhead for specific model size."""
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory before anything
    mem_before = measure_memory()
    
    model = create_model(model_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = get_model_size_bytes(model)
    
    mem_after_model = measure_memory()
    
    # Data
    X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    # ARC setup
    shard = None
    if use_arc:
        config = SelfHealingConfig(
            checkpoint_frequency=10,
            verbose=False,
        )
        shard = SelfHealingArc(model, optimizer, config)
    
    mem_after_arc = measure_memory()
    
    # Run training
    step_times = []
    step = 0
    
    start_total = time.time()
    
    for epoch in range(10):
        for batch_idx, (x, labels) in enumerate(dataloader):
            step += 1
            if step > n_steps:
                break
            
            step_start = time.time()
            
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, labels)
            
            if shard:
                action = shard.step(loss)
                if action.should_skip:
                    continue
            
            loss.backward()
            
            if shard:
                shard.post_backward()
            
            optimizer.step()
            
            step_times.append(time.time() - step_start)
        
        if step > n_steps:
            break
    
    total_time = time.time() - start_total
    mem_peak = measure_memory()
    
    return {
        'model_size': model_size,
        'n_params': n_params,
        'model_bytes_mb': model_bytes / 1e6,
        'use_arc': use_arc,
        'total_time_s': total_time,
        'steps': n_steps,
        'avg_step_time_ms': np.mean(step_times) * 1000,
        'std_step_time_ms': np.std(step_times) * 1000,
        'mem_model_mb': (mem_after_model - mem_before) / 1e6,
        'mem_arc_overhead_mb': (mem_after_arc - mem_after_model) / 1e6 if use_arc else 0,
        'mem_peak_mb': (mem_peak - mem_before) / 1e6,
    }


def run_overhead_analysis():
    """Run full overhead analysis."""
    
    print("="*70)
    print("OVERHEAD ANALYSIS")
    print("   Measuring time and memory overhead")
    print("="*70)
    
    sizes = ["small", "medium", "large", "xlarge"]
    
    all_results = []
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"MODEL SIZE: {size.upper()}")
        print("="*70)
        
        # Baseline (no ARC)
        baseline = run_overhead_test(size, use_arc=False)
        print(f"  Baseline: {baseline['n_params']/1e6:.2f}M params")
        print(f"    Time: {baseline['total_time_s']:.2f}s")
        print(f"    Step: {baseline['avg_step_time_ms']:.2f}ms ± {baseline['std_step_time_ms']:.2f}ms")
        print(f"    Memory: {baseline['mem_peak_mb']:.1f} MB")
        
        # With ARC
        arc = run_overhead_test(size, use_arc=True)
        print(f"  With ARC:")
        print(f"    Time: {arc['total_time_s']:.2f}s")
        print(f"    Step: {arc['avg_step_time_ms']:.2f}ms ± {arc['std_step_time_ms']:.2f}ms")
        print(f"    Memory: {arc['mem_peak_mb']:.1f} MB")
        print(f"    ARC overhead: {arc['mem_arc_overhead_mb']:.1f} MB")
        
        # Compute overheads
        time_overhead = (arc['total_time_s'] - baseline['total_time_s']) / baseline['total_time_s'] * 100
        mem_overhead = (arc['mem_peak_mb'] - baseline['mem_peak_mb']) / baseline['mem_peak_mb'] * 100 if baseline['mem_peak_mb'] > 0 else 0
        
        print(f"  Overhead:")
        print(f"    Time: {time_overhead:.1f}%")
        print(f"    Memory: {mem_overhead:.1f}%")
        
        all_results.append({
            'size': size,
            'baseline': baseline,
            'arc': arc,
            'time_overhead_pct': time_overhead,
            'mem_overhead_pct': mem_overhead,
        })
    
    # Summary table
    print("\n" + "="*70)
    print("OVERHEAD SUMMARY")
    print("="*70)
    
    print("\n| Size | Params | Time OH | Memory OH | ARC Mem |")
    print("|------|--------|---------|-----------|---------|")
    
    for r in all_results:
        print(f"| {r['size']:6} | {r['baseline']['n_params']/1e6:.1f}M | {r['time_overhead_pct']:6.1f}% | {r['mem_overhead_pct']:8.1f}% | {r['arc']['mem_arc_overhead_mb']:6.1f} MB |")
    
    # Save
    with open("overhead_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print("\nResults saved to: overhead_results.json")
    
    return all_results


if __name__ == "__main__":
    run_overhead_analysis()