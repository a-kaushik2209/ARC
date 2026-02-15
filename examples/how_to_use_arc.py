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
HOW TO USE ARC IN YOUR PROJECTS
================================

Just like YOLO uses CNN, you can use ARC with ANY neural network!

ARC is a WRAPPER - it goes AROUND your existing model and adds
self-monitoring, failure prediction, and automatic intervention.
"""

# =============================================================================
# USAGE PATTERN 1: Just like YOLO uses CNN
# =============================================================================

"""
YOLO Pattern:
    backbone = CNN()       # Feature extractor
    yolo = YOLO(backbone)  # Detection head on top
    output = yolo(image)   # Detects objects

ARC Pattern:
    model = YourModel()           # Any model (CNN, RNN, Transformer, PINN)
    arc = ArcV2.auto(model, opt)  # ARC wraps it
    arc.step(loss)                # ARC monitors & protects
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import ARC
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arc import ArcV2, Arc, Config


# =============================================================================
# EXAMPLE 1: Image Classification (Like using CNN in YOLO)
# =============================================================================

def example_image_classification():
    """
    Use ARC with a CNN for image classification.
    ARC will monitor training and prevent failures.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Image Classification with CNN")
    print("="*60)
    
    # Your CNN model (this could be ResNet, VGG, anything)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Create model and optimizer
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # === WRAP WITH ARC ===
    arc = ArcV2.auto(model, optimizer)
    
    # Training loop (same as normal, just add arc.step())
    for epoch in range(10):
        # Simulated batch
        images = torch.randn(32, 3, 32, 32)
        labels = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # === ARC MONITORS EVERYTHING ===
        status = arc.step(loss)
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}")


# =============================================================================
# EXAMPLE 2: Object Detection (Like YOLO)
# =============================================================================

def example_object_detection():
    """
    Use ARC with a YOLO-like object detection model.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Object Detection (YOLO-style)")
    print("="*60)
    
    # Simplified YOLO-like model
    class SimpleYOLO(nn.Module):
        def __init__(self, num_classes=80):
            super().__init__()
            # Backbone (like Darknet)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Detection head
            self.detection_head = nn.Conv2d(64, num_classes + 5, 1)
        
        def forward(self, x):
            features = self.backbone(x)
            return self.detection_head(features)
    
    model = SimpleYOLO()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # === ARC WRAPS YOLO ===
    arc = ArcV2.auto(model, optimizer, task_type="classification")
    
    for epoch in range(5):
        images = torch.randn(8, 3, 416, 416)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = outputs.mean()  # Simplified loss
        loss.backward()
        optimizer.step()
        
        arc.step(loss)
    
    print("  YOLO training monitored by ARC!")


# =============================================================================
# EXAMPLE 3: NLP / Transformer
# =============================================================================

def example_transformer():
    """
    Use ARC with a Transformer model for NLP tasks.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Transformer for NLP")
    print("="*60)
    
    # Simple Transformer encoder
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=10000, d_model=256, nhead=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                num_layers=2
            )
            self.classifier = nn.Linear(d_model, 2)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            return self.classifier(x.mean(dim=1))
    
    model = SimpleTransformer()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # === ARC MONITORS TRANSFORMER ===
    arc = ArcV2.auto(model, optimizer, task_type="classification")
    
    for epoch in range(5):
        tokens = torch.randint(0, 10000, (16, 128))
        labels = torch.randint(0, 2, (16,))
        
        optimizer.zero_grad()
        outputs = model(tokens)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        arc.step(loss)
    
    print("  Transformer training monitored by ARC!")


# =============================================================================
# EXAMPLE 4: Physics-Informed Neural Network (PINN)
# =============================================================================

def example_pinn():
    """
    Use ARC with a PINN for solving PDEs.
    ARC provides specialized stabilization for PINNs!
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Physics-Informed Neural Network")
    print("="*60)
    
    from arc import PINNStabilizer
    
    # PINN for solving heat equation
    class HeatPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64),  # (x, t) input
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)   # u(x,t) output
            )
        
        def forward(self, inputs):
            return self.net(inputs)
    
    model = HeatPINN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # === ARC WITH PINN STABILIZER ===
    stabilizer = PINNStabilizer(model, n_loss_terms=3)
    
    for epoch in range(10):
        # Create inputs with gradients
        inputs = torch.rand(100, 2, requires_grad=True)
        
        # Compute actual physics losses
        u = model(inputs)
        pde_loss = (u ** 2).mean()  # Simplified PDE residual
        bc_loss = ((u - 0) ** 2).mean()  # Boundary loss
        ic_loss = ((u - 1) ** 2).mean()  # Initial loss
        
        # === STABILIZED LOSS ===
        total_loss = stabilizer.get_stabilized_loss([pde_loss, bc_loss, ic_loss])
        
        optimizer.zero_grad()
        total_loss.backward()
        stabilizer.stabilize_step()  # Prevent chaos!
        optimizer.step()
        
        stabilizer.update(epoch, total_loss)
    
    print("  PINN training stabilized by ARC!")


# =============================================================================
# EXAMPLE 5: Continual Learning (Multiple Tasks)
# =============================================================================

def example_continual_learning():
    """
    Use ARC to learn multiple tasks without forgetting previous ones.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Continual Learning (No Forgetting)")
    print("="*60)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    optimizer = optim.Adam(model.parameters())
    
    # === ARC WITH CONTINUAL LEARNING ===
    arc = ArcV2.auto(model, optimizer, safety_level="standard")
    
    # Task 1: MNIST-like
    arc.begin_task("mnist")
    for epoch in range(5):
        x = torch.randn(32, 20)
        y = torch.randint(0, 5, (32,))
        
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        arc.step(loss)
    
    # Consolidate Task 1 (protect its knowledge!)
    dummy_loader = DataLoader(TensorDataset(torch.randn(50, 20), torch.randint(0, 5, (50,))), batch_size=32)
    arc.consolidate_task(dummy_loader)
    
    # Task 2: Fashion-MNIST-like
    arc.begin_task("fashion_mnist")
    for epoch in range(5):
        x = torch.randn(32, 20) * 2  # Different distribution
        y = torch.randint(0, 5, (32,))
        
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        
        # === ADD EWC LOSS TO PREVENT FORGETTING ===
        loss = loss + arc.get_ewc_loss()
        
        loss.backward()
        optimizer.step()
        arc.step(loss)
    
    print("  2 tasks learned without forgetting!")


# =============================================================================
# EXAMPLE 6: Uncertainty-Aware Predictions
# =============================================================================

def example_uncertainty():
    """
    Get predictions with confidence intervals.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Uncertainty-Aware Predictions")
    print("="*60)
    
    from arc import ConformalPredictor
    from torch.utils.data import DataLoader, TensorDataset
    
    model = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 5))
    
    # === CONFORMAL PREDICTOR ===
    cp = ConformalPredictor(model, alpha=0.1)  # 90% coverage
    
    # Calibrate on held-out data
    cal_loader = DataLoader(
        TensorDataset(torch.randn(100, 30), torch.randint(0, 5, (100,))),
        batch_size=32
    )
    cp.calibrate(cal_loader)
    
    # Make prediction with uncertainty
    test_input = torch.randn(1, 30)
    pred_set = cp.predict(test_input)
    
    print(f"  Prediction: {pred_set.prediction}")
    print(f"  Confidence: {pred_set.confidence:.2%}")
    print(f"  Prediction Set: {pred_set.set_members}")
    print(f"  Coverage Guarantee: {pred_set.coverage_target:.0%}")


# =============================================================================
# EXAMPLE 7: Adversarial Defense
# =============================================================================

def example_adversarial():
    """
    Detect adversarial inputs trying to fool your model.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Adversarial Defense")
    print("="*60)
    
    from arc import AdversarialDetector
    from torch.utils.data import DataLoader, TensorDataset
    
    model = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 10))
    
    # === FIT DETECTOR ON CLEAN DATA ===
    detector = AdversarialDetector(model)
    clean_loader = DataLoader(
        TensorDataset(torch.randn(200, 50), torch.randint(0, 10, (200,))),
        batch_size=32
    )
    detector.fit(clean_loader)
    
    # Check if input is adversarial
    clean_input = torch.randn(1, 50)
    suspicious_input = torch.randn(1, 50) * 10  # Unusual input
    
    alert_clean = detector.detect(clean_input)
    alert_sus = detector.detect(suspicious_input)
    
    print(f"  Clean input - Adversarial: {alert_clean.is_adversarial}")
    print(f"  Suspicious input - Adversarial: {alert_sus.is_adversarial}")


# =============================================================================
# THE SIMPLEST POSSIBLE USAGE
# =============================================================================

def simplest_usage():
    """
    The absolute minimum code to use ARC.
    """
    print("\n" + "="*60)
    print("SIMPLEST USAGE - 3 LINES!")
    print("="*60)
    
    model = nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters())
    
    # === 1. CREATE ARC ===
    arc = ArcV2.auto(model, optimizer)
    
    # === 2. TRAIN AS NORMAL ===
    for epoch in range(5):
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        
        # === 3. JUST ADD THIS LINE ===
        arc.step(loss)
    
    print("  That's it! ARC is now protecting your training!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HOW TO USE ARC - COMPLETE EXAMPLES")
    print("="*60)
    
    simplest_usage()
    example_image_classification()
    example_object_detection()
    example_transformer()
    example_pinn()
    example_continual_learning()
    example_uncertainty()
    example_adversarial()
    
    print("\n" + "="*60)
    print("SUMMARY: ARC works with ANY neural network!")
    print("="*60)
    print("""
    CNN          → ArcV2.auto(cnn, optimizer)
    YOLO         → ArcV2.auto(yolo, optimizer)  
    Transformer  → ArcV2.auto(transformer, optimizer)
    PINN         → PINNStabilizer(pinn)
    RNN/LSTM     → ArcV2.auto(rnn, optimizer)
    GAN          → ArcV2.auto(generator, optimizer)
    
    Just add arc.step(loss) to your training loop!
    """)