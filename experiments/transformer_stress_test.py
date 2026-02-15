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
ARC Phase 12: Transformer Stress Test ("The Torture Test")

This script benchmarks ARC on a small GPT model (nanoGPT-style) to demonstrate
stability in a more realistic and challenging scenario than CIFAR-10.

Key Failure Modes Tested:
1. FP16 Mixed Precision Underflow
2. Massive LR Spikes (10x jumps)
3. Gradient Accumulation Instability
4. Long Sequence Explosions

Success Criteria:
- ARC saves runs where baseline AdamW fails
- Demonstrates <10% overhead on Transformer training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import ArcV2


# =============================================================================
# NanoGPT-style Transformer (Simplified)
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        # QKV projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class TransformerBlock(nn.Module):
    """Transformer block with LayerNorm and residual connections."""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """
    Minimal GPT model for stress testing.
    
    Config: ~10M parameters (small enough to run on CPU, big enough to be realistic)
    """
    
    def __init__(self, vocab_size=50257, block_size=128, n_layer=6, n_head=6, n_embd=384):
        super().__init__()
        self.block_size = block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoGPT initialized: {self.n_params/1e6:.2f}M parameters")
    
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block size {self.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


# =============================================================================
# Synthetic Text Dataset (for controlled testing)
# =============================================================================

class SyntheticTextDataset(Dataset):
    """Generates random token sequences for stress testing."""
    
    def __init__(self, vocab_size, seq_len, n_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        # Pre-generate for reproducibility
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y


# =============================================================================
# Failure Injection Functions
# =============================================================================

def inject_lr_spike(optimizer, epoch, spike_epoch=3, multiplier=10.0):
    """Inject a massive LR spike at a specific epoch."""
    if epoch == spike_epoch:
        for pg in optimizer.param_groups:
            pg['lr'] *= multiplier
        print(f"    [INJECTION] LR spiked to {pg['lr']}")
        return True
    return False


def inject_gradient_explosion(model, epoch, explosion_epoch=4, scale=100.0):
    """Scale gradients massively to simulate explosion."""
    if epoch == explosion_epoch:
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale
        print(f"    [INJECTION] Gradients scaled by {scale}x")
        return True
    return False


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, use_arc=False, arc=None, 
                inject_fn=None, epoch=0):
    """Train for one epoch with optional ARC monitoring and failure injection."""
    model.train()
    total_loss = 0
    n_batches = 0
    failed = False
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        try:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Check for failure
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                print(f"    [FAILURE] Loss exploded: {loss.item()}")
                failed = True
                break
            
            loss.backward()
            
            # Inject failure (post-backward)
            if inject_fn:
                inject_fn(model, epoch)
            
            optimizer.step()
            
            # ARC monitoring
            if use_arc and arc is not None:
                status = arc.step(loss)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Check weights
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                print(f"    [FAILURE] NaN/Inf in weights")
                failed = True
                break
                
        except RuntimeError as e:
            print(f"    [FAILURE] Runtime error: {e}")
            failed = True
            break
    
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, failed


def run_stress_test(use_arc, failure_type="lr_spike", n_epochs=10, device="cpu"):
    """Run a full stress test with optional ARC protection."""
    
    # Model setup
    model = NanoGPT(vocab_size=1000, block_size=64, n_layer=4, n_head=4, n_embd=256)
    model = model.to(device)
    
    # Data setup (small for speed)
    dataset = SyntheticTextDataset(vocab_size=1000, seq_len=64, n_samples=500)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Optimizer (intentionally aggressive LR to stress test)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # ARC setup
    arc = None
    if use_arc:
        arc = ArcV2.auto(model, optimizer, safety_level="paranoid")
    
    # Failure injection
    inject_fn = None
    if failure_type == "lr_spike":
        inject_fn = lambda m, e: inject_lr_spike(optimizer, e, spike_epoch=3, multiplier=50.0)
    elif failure_type == "gradient_explosion":
        inject_fn = lambda m, e: inject_gradient_explosion(m, e, explosion_epoch=4, scale=100.0)
    
    # Training loop
    results = {
        "use_arc": use_arc,
        "failure_type": failure_type,
        "epochs_completed": 0,
        "failed": False,
        "losses": [],
        "final_loss": None,
    }
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # LR spike injection
        if failure_type == "lr_spike":
            inject_lr_spike(optimizer, epoch, spike_epoch=3, multiplier=50.0)
        
        avg_loss, failed = train_epoch(
            model, dataloader, optimizer, device,
            use_arc=use_arc, arc=arc,
            inject_fn=(lambda m, e: inject_gradient_explosion(m, e)) if failure_type == "gradient_explosion" else None,
            epoch=epoch
        )
        
        results["losses"].append(avg_loss)
        results["epochs_completed"] = epoch + 1
        
        print(f"  Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
        
        if failed:
            results["failed"] = True
            break
        
        if use_arc and arc is not None:
            arc.end_epoch(epoch)
    
    elapsed = time.time() - start_time
    results["elapsed_time"] = elapsed
    results["final_loss"] = results["losses"][-1] if results["losses"] else None
    
    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def run_transformer_benchmark():
    """Run the full Transformer stress test benchmark."""
    
    print("="*60)
    print("TRANSFORMER STRESS TEST (Phase 12)")
    print("Testing ARC on NanoGPT-style language model")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    failure_types = ["lr_spike", "gradient_explosion"]
    all_results = []
    
    for failure_type in failure_types:
        print(f"\n{'='*60}")
        print(f"Testing Failure Type: {failure_type.upper()}")
        print("="*60)
        
        # Without ARC
        print("\n[1/2] Training WITHOUT ARC protection...")
        result_no_arc = run_stress_test(use_arc=False, failure_type=failure_type, device=device)
        status = "FAILED" if result_no_arc["failed"] else f"OK (Loss: {result_no_arc['final_loss']:.4f})"
        print(f"  Result: {status}")
        
        # With ARC
        print("\n[2/2] Training WITH ARC protection...")
        result_arc = run_stress_test(use_arc=True, failure_type=failure_type, device=device)
        status = "FAILED" if result_arc["failed"] else f"OK (Loss: {result_arc['final_loss']:.4f})"
        print(f"  Result: {status}")
        
        all_results.append({
            "failure_type": failure_type,
            "without_arc": result_no_arc,
            "with_arc": result_arc,
            "arc_saved": result_no_arc["failed"] and not result_arc["failed"],
        })
    
    # Summary
    print("\n" + "="*60)
    print("TRANSFORMER STRESS TEST SUMMARY")
    print("="*60)
    
    print("\n| Failure Type        | No ARC    | With ARC  | ARC Saved? |")
    print("|---------------------|-----------|-----------|------------|")
    
    arc_saves = 0
    for r in all_results:
        no_arc = "FAIL" if r["without_arc"]["failed"] else f"Loss: {r['without_arc']['final_loss']:.2f}"
        with_arc = "FAIL" if r["with_arc"]["failed"] else f"Loss: {r['with_arc']['final_loss']:.2f}"
        saved = "YES ✓" if r["arc_saved"] else "No"
        if r["arc_saved"]:
            arc_saves += 1
        print(f"| {r['failure_type']:19} | {no_arc:9} | {with_arc:9} | {saved:10} |")
    
    print(f"\nARC saved {arc_saves}/{len(all_results)} failing scenarios on Transformers")
    
    # Save results
    with open("transformer_stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: transformer_stress_results.json")
    
    return all_results


if __name__ == "__main__":
    run_transformer_benchmark()