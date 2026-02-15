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
ARC Phase 13: FP16 Mixed-Precision Underflow Stress Test

This test targets REAL production failure modes in distributed training:
1. FP16 underflow (gradients become zero)
2. FP16 overflow (gradients become inf)
3. Loss scaling failures

These are the silent killers of LLM training runs that standard gradient
clipping cannot detect (because the values are "valid" but useless).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import json
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import ArcV2


# =============================================================================
# Simple Transformer for FP16 Testing
# =============================================================================

class SimpleTransformer(nn.Module):
    """Minimal transformer prone to FP16 issues."""
    
    def __init__(self, vocab_size=1000, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model) * 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleTransformer: {self.n_params/1e6:.2f}M params")
    
    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x) + self.pos_encoding[:, :T, :]
        x = self.transformer(x)
        return self.fc(x)


# =============================================================================
# Synthetic Dataset
# =============================================================================

class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, n_samples):
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


# =============================================================================
# FP16 Failure Injection
# =============================================================================

def inject_underflow(model, scale=1e-8):
    """Scale weights to cause underflow in FP16."""
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    print(f"  [INJECTION] Weights scaled by {scale} (underflow)")


def inject_overflow(model, scale=1e4):
    """Scale weights to cause overflow in FP16."""
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    print(f"  [INJECTION] Weights scaled by {scale} (overflow)")


def inject_loss_spike(loss_value, spike_factor=1e6):
    """Artificially spike the loss to trigger scaler issues."""
    return loss_value * spike_factor


# =============================================================================
# Training Functions
# =============================================================================

def train_with_fp16(model, dataloader, optimizer, device, n_epochs=5,
                    use_arc=False, arc=None, failure_type=None, failure_epoch=2):
    """Train with mixed precision and optional failure injection."""
    
    scaler = GradScaler()
    results = {
        "epochs_completed": 0,
        "failed": False,
        "losses": [],
        "scaler_scale": [],
        "failure_type": failure_type,
    }
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Inject failure at specific epoch
        if failure_type == "underflow" and epoch == failure_epoch:
            inject_underflow(model, scale=1e-7)
        elif failure_type == "overflow" and epoch == failure_epoch:
            inject_overflow(model, scale=1e4)
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            try:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Loss spike injection
                    if failure_type == "loss_spike" and epoch == failure_epoch and n_batches == 0:
                        loss = inject_loss_spike(loss)
                
                # Check for failure
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [FAILURE] NaN/Inf loss at epoch {epoch}")
                    results["failed"] = True
                    break
                
                # Scaled backward
                scaler.scale(loss).backward()
                
                # Check for zero gradients (underflow)
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                
                if grad_norm == 0:
                    print(f"  [FAILURE] Zero gradients (underflow)")
                    results["failed"] = True
                    break
                
                scaler.step(optimizer)
                scaler.update()
                
                # Check scaler state
                scale = scaler.get_scale()
                if scale < 1e-10:
                    print(f"  [FAILURE] Scaler collapsed to {scale}")
                    results["failed"] = True
                    break
                
                # ARC monitoring
                if use_arc and arc:
                    arc.step(loss.item())
                
                epoch_loss += loss.item()
                n_batches += 1
                
            except RuntimeError as e:
                if "inf" in str(e).lower() or "nan" in str(e).lower():
                    print(f"  [FAILURE] Runtime: {str(e)[:50]}")
                    results["failed"] = True
                    break
                raise
        
        if results["failed"]:
            break
        
        avg_loss = epoch_loss / max(n_batches, 1)
        results["losses"].append(avg_loss)
        results["scaler_scale"].append(scaler.get_scale())
        results["epochs_completed"] = epoch + 1
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Scale={scaler.get_scale():.1e}")
        
        if use_arc and arc:
            arc.end_epoch(epoch)
    
    return results


def run_fp16_stress_test():
    """Run FP16 mixed precision stress tests."""
    
    print("="*60)
    print("FP16 MIXED PRECISION STRESS TEST (Phase 13)")
    print("Testing real production failure modes")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cpu":
        print("\nWARNING: FP16 stress test is most relevant on GPU.")
        print("   Running in CPU mode with simulated failures.\n")
    
    # Data setup
    dataset = SyntheticTextDataset(vocab_size=1000, seq_len=64, n_samples=200)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    failure_types = ["underflow", "overflow", "loss_spike", None]
    all_results = []
    
    for failure_type in failure_types:
        label = failure_type or "baseline"
        print(f"\n{'='*60}")
        print(f"Testing: {label.upper()}")
        print("="*60)
        
        # Without ARC
        print("\n[1/2] Training WITHOUT ARC...")
        model = SimpleTransformer().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        result_no_arc = train_with_fp16(
            model, dataloader, optimizer, device,
            n_epochs=5, use_arc=False, failure_type=failure_type
        )
        status = "FAILED" if result_no_arc["failed"] else f"OK (Loss: {result_no_arc['losses'][-1]:.4f})"
        print(f"  Result: {status}")
        
        # With ARC
        print("\n[2/2] Training WITH ARC...")
        model = SimpleTransformer().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        arc = ArcV2.auto(model, optimizer, safety_level="paranoid")
        
        result_arc = train_with_fp16(
            model, dataloader, optimizer, device,
            n_epochs=5, use_arc=True, arc=arc, failure_type=failure_type
        )
        status = "FAILED" if result_arc["failed"] else f"OK (Loss: {result_arc['losses'][-1]:.4f})"
        print(f"  Result: {status}")
        
        all_results.append({
            "failure_type": label,
            "without_arc": result_no_arc,
            "with_arc": result_arc,
            "arc_saved": result_no_arc["failed"] and not result_arc["failed"],
        })
    
    # Summary
    print("\n" + "="*60)
    print("FP16 STRESS TEST SUMMARY")
    print("="*60)
    
    print("\n| Failure Type     | No ARC    | With ARC  | ARC Saved? |")
    print("|------------------|-----------|-----------|------------|")
    
    arc_saves = 0
    for r in all_results:
        no_arc = "FAIL" if r["without_arc"]["failed"] else "OK"
        with_arc = "FAIL" if r["with_arc"]["failed"] else "OK"
        saved = "YES ✓" if r["arc_saved"] else "No"
        if r["arc_saved"]:
            arc_saves += 1
        print(f"| {r['failure_type']:16} | {no_arc:9} | {with_arc:9} | {saved:10} |")
    
    print(f"\nARC saved {arc_saves}/{len(all_results)} failing scenarios")
    
    with open("fp16_stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: fp16_stress_results.json")
    
    return all_results


if __name__ == "__main__":
    run_fp16_stress_test()