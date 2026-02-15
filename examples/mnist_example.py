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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arc import Arc, Config


class SimpleCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer, prophet):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        prophet.on_batch_end(loss.item())
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    
    return val_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Arc MNIST Example')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--induce-failure', action='store_true',
                       help='Artificially increase LR to cause divergence')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    except Exception as e:
        print(f"Could not load MNIST dataset: {e}")
        print("Creating synthetic data for demonstration...")
        train_dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 1, 28, 28),
            torch.randint(0, 10, (1000,))
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.randn(200, 1, 28, 28),
            torch.randint(0, 10, (200,))
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    config = Config()
    config.signal.activation_sample_ratio = 0.2
    
    prophet = Arc(config=config, verbose=True)
    prophet.attach(model, optimizer)
    
    print(f"\n{'='*60}")
    print("Starting training with Arc monitoring")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        if args.induce_failure and epoch == 10:
            print("\nInducing failure: Increasing learning rate 100x\n")
            for g in optimizer.param_groups:
                g['lr'] *= 100
        
        train_loss = train_epoch(model, device, train_loader, optimizer, prophet)
        
        val_loss, val_acc = validate(model, device, val_loader)
        
        prediction = prophet.on_epoch_end(epoch, val_loss=val_loss)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2%}")
        print(f"  Risk Level: {prediction.risk_level.upper()}")
        
        if prediction.risk_level in ['high', 'critical']:
            mode, prob = prediction.get_highest_risk_mode()
            print(f"   {mode}: {prob:.1%}")
            print(f"  Recommendation: {prediction.recommendation.action}")
    
    prophet.detach()
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Final overhead: {prophet.overhead_percentage:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()