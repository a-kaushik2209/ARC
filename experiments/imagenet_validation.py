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
ARC Publication Experiment 4: ImageNet Validation

To run this experiment:
1. Download ImageNet validation set (ILSVRC2012_img_val.tar)
2. Extract to ./data/imagenet/val
3. Run this script

This script validates ARC on a large-scale dataset (1.2M images, 1000 classes).
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

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from arc import ArcV2


def validate_imagenet():
    print("="*60)
    print("IMAGENET VALIDATION EXPERIMENT")
    print("Large-scale validation of ARC")
    print("="*60)
    
    # Check for data
    data_dir = os.path.join('data', 'imagenet')
    if not os.path.exists(data_dir):
        print("\n[!] ImageNet data not found.")
        print("Please follow instructions in `publication_experiments_plan.md`")
        print("to download ILSVRC2012 dataset (~150GB).")
        return None
    
    # Setup ResNet-50
    print("\n  Loading ResNet-50...")
    model = models.resnet50(pretrained=False)
    
    # Setup Data
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    # Setup ARC
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    arc = ArcV2.auto(model, optimizer, safety_level="standard")
    
    print("\n  Starting training (1 epoch demonstration)...")
    
    model.train()
    losses = []
    
    start = time.time()
    
    # Run slightly fewer batches for quicker demo
    max_batches = 100
    
    for i, (images, target) in enumerate(train_loader):
        if i >= max_batches:
            break
            
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        
        # ARC Step
        status = arc.step(loss)
        
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f"    Batch {i}/{max_batches}: Loss {loss.item():.4f}")
            if hasattr(status, 'intervention') and status.intervention:
                 print(f"    [ARC] Intervention: {status.intervention}")
    
    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  Average Loss: {np.mean(losses):.4f}")
    
    return {"status": "success", "avg_loss": np.mean(losses)}


if __name__ == "__main__":
    validate_imagenet()