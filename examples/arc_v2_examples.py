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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from arc import ArcV2, Config

def example_basic_usage():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    optimizer = optim.Adam(model.parameters())
    
    arc = ArcV2.auto(model, optimizer)
    
    for epoch in range(10):
        for batch in range(100):
            x = torch.randn(32, 784)
            y = torch.randint(0, 10, (32,))
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            
            status = arc.step(loss)
        
        result = arc.end_epoch(epoch)
        print(f"Epoch {epoch}: Risk Level = {result['risk_level']}")
    
    print("\n" + arc.health_report())

def example_continual_learning():
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    optimizer = optim.Adam(model.parameters())
    
    arc = ArcV2.auto(model, optimizer, safety_level="standard")
    
    arc.begin_task("mnist")
    
    for epoch in range(5):
        loss = torch.tensor(1.0 - epoch * 0.15)
        arc.step(loss)
        arc.end_epoch(epoch)
    
    dummy_data = TensorDataset(torch.randn(100, 784), torch.randint(0, 10, (100,)))
    dummy_loader = DataLoader(dummy_data, batch_size=32)
    
    arc.consolidate_task(dummy_loader)
    
    arc.begin_task("fashion_mnist")
    
    for epoch in range(5):
        loss = torch.tensor(1.0 - epoch * 0.1)
        
        ewc_loss = arc.get_ewc_loss()
        total_loss = loss + ewc_loss
        
        arc.step(total_loss)
        arc.end_epoch(epoch)
    
    print("Continual learning complete!")
    print(f"Tasks consolidated: {arc._ewc.n_tasks}")

def example_uncertainty():
    
    from arc import ConformalPredictor
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    cp = ConformalPredictor(model, alpha=0.1)
    
    cal_data = TensorDataset(torch.randn(200, 784), torch.randint(0, 10, (200,)))
    cal_loader = DataLoader(cal_data, batch_size=32)
    
    threshold = cp.calibrate(cal_loader)
    print(f"Calibration threshold: {threshold:.4f}")
    
    test_input = torch.randn(1, 784)
    pred_set = cp.predict(test_input)
    
    print(f"\nPrediction: {pred_set.prediction}")
    print(f"Confidence: {pred_set.confidence:.2%}")
    print(f"Prediction Set: {pred_set.set_members}")
    print(f"Set Size: {pred_set.set_size}")
    print(f"Coverage Target: {pred_set.coverage_target:.0%}")

def example_adversarial():
    
    from arc import AdversarialDetector, AdversarialTrainer
    
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    detector = AdversarialDetector(model)
    
    clean_data = TensorDataset(torch.randn(500, 784), torch.randint(0, 10, (500,)))
    clean_loader = DataLoader(clean_data, batch_size=32)
    
    detector.fit(clean_loader, n_samples=500)
    print("Adversarial detector fitted!")
    
    clean_input = torch.randn(1, 784)
    alert = detector.detect(clean_input)
    
    print(f"\nClean input detection:")
    print(f"  Is adversarial: {alert.is_adversarial}")
    print(f"  Confidence: {alert.confidence:.2%}")
    print(f"  Recommendation: {alert.recommendation}")
    
    trainer = AdversarialTrainer(model, attack="pgd", epsilon=0.1)
    optimizer = optim.Adam(model.parameters())
    
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    clean_loss, adv_loss = trainer.adversarial_train_step(x, y, optimizer)
    print(f"\nAdversarial training step:")
    print(f"  Clean loss: {clean_loss:.4f}")
    print(f"  Adversarial loss: {adv_loss:.4f}")

def example_pinn():
    
    from arc import PINNStabilizer
    
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    stabilizer = PINNStabilizer(model, n_loss_terms=3)
    
    for epoch in range(50):
        params = stabilizer.get_training_params()
        freq_scale = params["frequency_scale"]
        
        pde_loss = torch.tensor(1.0 / (epoch + 1) * freq_scale)
        bc_loss = torch.tensor(0.5 / (epoch + 1))
        ic_loss = torch.tensor(0.3 / (epoch + 1))
        
        total_loss = stabilizer.get_stabilized_loss([pde_loss, bc_loss, ic_loss])
        
        optimizer.zero_grad()
        total_loss.backward()
        
        stabilizer.stabilize_step()
        
        optimizer.step()
        
        diagnostics = stabilizer.update(epoch, total_loss, [
            pde_loss.item(), bc_loss.item(), ic_loss.item()
        ])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Stability Score: {diagnostics.stability_score:.2f}")
            print(f"  Loss Balance: {diagnostics.loss_balance}")
            print(f"  Recommendation: {diagnostics.recommendation}")

def example_spectral():
    
    from arc import FourierFeatureEncoder, SpectralAnalyzer
    
    encoder = FourierFeatureEncoder(
        input_dim=2,
        n_frequencies=128,
        sigma=10.0
    )
    
    model = nn.Sequential(
        encoder,
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    x = torch.rand(100, 2)
    encoded = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    
    analyzer = SpectralAnalyzer()
    
    t = torch.linspace(0, 10, 1000)
    signal = torch.sin(t) + 0.5 * torch.sin(5*t) + 0.1 * torch.sin(20*t)
    
    analysis = analyzer.analyze(signal)
    print(f"\nSpectral Analysis:")
    print(f"  Spectral Entropy: {analysis.spectral_entropy:.2f}")
    print(f"  High Freq Ratio: {analysis.high_freq_ratio:.2%}")
    print(f"  Recommendation: {analysis.recommendation}")

if __name__ == "__main__":
    print("=" * 60)
    print("ARC v2.0 - Complete Usage Examples")
    print("=" * 60)
    
    print("\n1. BASIC USAGE")
    print("-" * 40)
    example_basic_usage()
    
    print("\n\n2. CONTINUAL LEARNING")
    print("-" * 40)
    example_continual_learning()
    
    print("\n\n3. UNCERTAINTY QUANTIFICATION")
    print("-" * 40)
    example_uncertainty()
    
    print("\n\n4. ADVERSARIAL ROBUSTNESS")
    print("-" * 40)
    example_adversarial()
    
    print("\n\n5. PINN STABILIZATION")
    print("-" * 40)
    example_pinn()
    
    print("\n\n6. SPECTRAL FEATURES")
    print("-" * 40)
    example_spectral()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)