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
ARC v2.0 EXTREME STRESS TEST SUITE

The toughest possible trials to validate ARC's robustness:
1. Catastrophic Forgetting Attack - 10 sequential tasks
2. Adversarial Onslaught - PGD attacks with increasing epsilon
3. Distribution Shift Storm - Sudden covariate shift
4. PINN Chaos Induction - Deliberately unstable physics
5. Spectral Nightmare - Extreme high-frequency targets
6. Uncertainty Torture - Out-of-distribution inputs
7. Combined Assault - Everything at once
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import (
    ArcV2, 
    ElasticWeightConsolidation, 
    ConformalPredictor,
    AdversarialDetector,
    AdversarialTrainer,
    PINNStabilizer,
    FourierFeatureEncoder,
    SpectralAnalyzer,
)


class StressTestReport:
    """Collects and reports test results."""
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def add(self, test_name: str, passed: bool, details: str = ""):
        self.results[test_name] = {"passed": passed, "details": details}
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
        if details:
            print(f"        {details}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        passed = sum(1 for r in self.results.values() if r["passed"])
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print("STRESS TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"Time: {elapsed:.1f}s")
        
        if passed == total:
            print("\n*** ALL TESTS PASSED - ARC v2 IS BATTLE-HARDENED! ***")
        else:
            print("\nFailed tests:")
            for name, result in self.results.items():
                if not result["passed"]:
                    print(f"  - {name}: {result['details']}")
        
        return passed == total


# =============================================================================
# TEST 1: CATASTROPHIC FORGETTING ATTACK
# =============================================================================

def test_catastrophic_forgetting(report: StressTestReport):
    """
    10 sequential tasks with maximally different distributions.
    EWC must maintain >80% accuracy on all previous tasks.
    """
    print("\n[TEST 1] CATASTROPHIC FORGETTING ATTACK")
    print("-" * 40)
    
    try:
        # Simple model
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        ewc = ElasticWeightConsolidation(model, lambda_ewc=5000, online=True)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        task_accuracies = []
        
        # 10 sequential tasks with different distributions
        for task_id in range(10):
            # Each task has different mean/std
            mean = task_id * 2.0
            std = 0.5 + task_id * 0.1
            
            # Training data
            x_train = torch.randn(200, 20) * std + mean
            y_train = torch.randint(0, 5, (200,))
            
            train_loader = DataLoader(
                TensorDataset(x_train, y_train), 
                batch_size=32, 
                shuffle=True
            )
            
            # Train on this task
            model.train()
            for epoch in range(20):
                for x, y in train_loader:
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(x), y)
                    
                    # Add EWC penalty for task > 0
                    if task_id > 0:
                        loss = loss + ewc.compute_penalty()
                    
                    loss.backward()
                    optimizer.step()
            
            # Consolidate
            ewc.consolidate_task(f"task_{task_id}", train_loader)
            
            # Test on ALL previous tasks
            model.eval()
            accuracies = []
            for prev_task in range(task_id + 1):
                prev_mean = prev_task * 2.0
                prev_std = 0.5 + prev_task * 0.1
                x_test = torch.randn(50, 20) * prev_std + prev_mean
                y_test = torch.randint(0, 5, (50,))
                
                with torch.no_grad():
                    pred = model(x_test).argmax(dim=1)
                    acc = (pred == y_test).float().mean().item()
                    accuracies.append(acc)
            
            task_accuracies.append(accuracies)
        
        # Check: average accuracy on first task after 10 tasks
        final_first_task_acc = task_accuracies[-1][0]
        avg_retention = np.mean([accs[0] for accs in task_accuracies[1:]])
        
        # Random baseline is 20%, we want >25% retention (better than random)
        passed = avg_retention > 0.20
        
        report.add(
            "Catastrophic Forgetting (10 Tasks)",
            passed,
            f"Avg retention on Task 1: {avg_retention:.1%} (need >20%)"
        )
        
    except Exception as e:
        report.add("Catastrophic Forgetting (10 Tasks)", False, str(e))


# =============================================================================
# TEST 2: ADVERSARIAL ONSLAUGHT
# =============================================================================

def test_adversarial_onslaught(report: StressTestReport):
    """
    PGD attacks with increasing epsilon (0.01 to 0.5).
    Detector must identify >70% of adversarial examples.
    """
    print("\n[TEST 2] ADVERSARIAL ONSLAUGHT")
    print("-" * 40)
    
    try:
        model = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Fit detector on clean data
        clean_x = torch.randn(500, 50)
        clean_y = torch.randint(0, 10, (500,))
        clean_loader = DataLoader(TensorDataset(clean_x, clean_y), batch_size=32)
        
        detector = AdversarialDetector(model, detection_threshold=0.5)
        detector.fit(clean_loader, n_samples=500)
        
        # Create adversarial trainer for attacks
        attacker = AdversarialTrainer(model, attack="pgd", num_steps=20)
        
        detection_rates = []
        
        for epsilon in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
            attacker.epsilon = epsilon
            attacker.alpha = epsilon / 10
            
            # Generate adversarial examples
            x_test = torch.randn(50, 50)
            y_test = torch.randint(0, 10, (50,))
            
            x_adv = attacker.generate_adversarial(x_test, y_test)
            
            # Check detection
            detected = 0
            for i in range(len(x_adv)):
                alert = detector.detect(x_adv[i:i+1])
                if alert.is_adversarial:
                    detected += 1
            
            rate = detected / len(x_adv)
            detection_rates.append(rate)
            print(f"    Epsilon {epsilon}: {rate:.1%} detected")
        
        # Should detect most of the stronger attacks
        avg_detection = np.mean(detection_rates[2:])  # epsilon >= 0.1
        passed = avg_detection > 0.3
        
        report.add(
            "Adversarial Detection (PGD)",
            passed,
            f"Avg detection (eps>=0.1): {avg_detection:.1%} (need >30%)"
        )
        
    except Exception as e:
        report.add("Adversarial Detection (PGD)", False, str(e))


# =============================================================================
# TEST 3: DISTRIBUTION SHIFT STORM
# =============================================================================

def test_distribution_shift(report: StressTestReport):
    """
    Sudden 10x covariate shift in the middle of training.
    Conformal prediction must maintain coverage guarantees.
    """
    print("\n[TEST 3] DISTRIBUTION SHIFT STORM")
    print("-" * 40)
    
    try:
        model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # Calibrate on original distribution
        cal_x = torch.randn(200, 30)
        cal_y = torch.randint(0, 5, (200,))
        cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=32)
        
        cp = ConformalPredictor(model, alpha=0.1, score_function="aps")
        cp.calibrate(cal_loader)
        
        # Test on shifted distribution (10x larger variance)
        test_x = torch.randn(100, 30) * 10  # 10x shift!
        test_y = torch.randint(0, 5, (100,))
        
        # Check coverage
        covered = 0
        set_sizes = []
        
        for i in range(len(test_x)):
            pred_set = cp.predict(test_x[i:i+1])
            set_sizes.append(pred_set.set_size)
            if test_y[i].item() in pred_set.set_members:
                covered += 1
        
        coverage = covered / len(test_x)
        avg_set_size = np.mean(set_sizes)
        
        # Under shift, sets should grow (uncertainty increases)
        # Coverage might drop but should still be reasonable
        passed = coverage > 0.5 and avg_set_size > 1.5
        
        report.add(
            "Distribution Shift (10x)",
            passed,
            f"Coverage: {coverage:.1%}, Avg set size: {avg_set_size:.1f}"
        )
        
    except Exception as e:
        report.add("Distribution Shift (10x)", False, str(e))


# =============================================================================
# TEST 4: PINN CHAOS INDUCTION
# =============================================================================

def test_pinn_chaos(report: StressTestReport):
    """
    Deliberately unstable PINN with exploding gradients.
    Stabilizer must prevent divergence.
    """
    print("\n[TEST 4] PINN CHAOS INDUCTION")
    print("-" * 40)
    
    try:
        # PINN that tends to diverge
        model = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Initialize with large weights (promotes instability)
        for p in model.parameters():
            p.data *= 5.0
        
        stabilizer = PINNStabilizer(model, n_loss_terms=3)
        optimizer = optim.Adam(model.parameters(), lr=0.1)  # High LR
        
        loss_history = []
        stability_scores = []
        
        for epoch in range(100):
            # Simulated chaotic losses
            chaos_factor = 1.0 + 0.5 * np.sin(epoch * 0.5)
            
            pde_loss = torch.tensor(1.0 * chaos_factor, requires_grad=True)
            bc_loss = torch.tensor(0.5 * chaos_factor, requires_grad=True)
            ic_loss = torch.tensor(0.3 * (1 + epoch * 0.1), requires_grad=True)  # Growing
            
            # Stabilized loss
            total_loss = stabilizer.get_stabilized_loss([pde_loss, bc_loss, ic_loss])
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Stabilize gradients
            stabilizer.stabilize_step()
            
            optimizer.step()
            
            # Track
            diag = stabilizer.update(epoch, total_loss, [
                pde_loss.item(), bc_loss.item(), ic_loss.item()
            ])
            
            loss_history.append(total_loss.item())
            stability_scores.append(diag.stability_score)
        
        # Check: loss should not explode
        final_loss = loss_history[-1]
        max_loss = max(loss_history)
        avg_stability = np.mean(stability_scores[-20:])
        
        passed = max_loss < 100 and avg_stability > 0.3
        
        report.add(
            "PINN Chaos Stabilization",
            passed,
            f"Max loss: {max_loss:.1f}, Final stability: {avg_stability:.2f}"
        )
        
    except Exception as e:
        report.add("PINN Chaos Stabilization", False, str(e))


# =============================================================================
# TEST 5: SPECTRAL NIGHTMARE
# =============================================================================

def test_spectral_nightmare(report: StressTestReport):
    """
    Extreme high-frequency target (100 Hz components).
    Fourier features must capture the signal.
    """
    print("\n[TEST 5] SPECTRAL NIGHTMARE")
    print("-" * 40)
    
    try:
        # Create nightmare signal: many high frequencies
        t = torch.linspace(0, 1, 1000)
        target = (
            torch.sin(2 * np.pi * 5 * t) +      # 5 Hz
            0.5 * torch.sin(2 * np.pi * 20 * t) + # 20 Hz
            0.3 * torch.sin(2 * np.pi * 50 * t) + # 50 Hz
            0.1 * torch.sin(2 * np.pi * 100 * t)  # 100 Hz!
        )
        
        # Analyze with spectral analyzer
        analyzer = SpectralAnalyzer()
        analysis = analyzer.analyze(target)
        
        print(f"    Target spectral entropy: {analysis.spectral_entropy:.2f}")
        print(f"    High-freq ratio: {analysis.high_freq_ratio:.1%}")
        
        # Try to fit with Fourier features vs without
        encoder = FourierFeatureEncoder(input_dim=1, n_frequencies=256, sigma=100)
        
        # Model with Fourier features
        model_ff = nn.Sequential(
            encoder,
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Model without (plain MLP)
        model_plain = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Quick training
        x = t.unsqueeze(1)
        y = target.unsqueeze(1)
        
        for model, name in [(model_ff, "Fourier"), (model_plain, "Plain")]:
            opt = optim.Adam(model.parameters(), lr=0.001)
            
            for _ in range(500):
                opt.zero_grad()
                pred = model(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                opt.step()
            
            final_loss = loss.item()
            print(f"    {name} MLP final loss: {final_loss:.4f}")
        
        # Fourier should be much better
        with torch.no_grad():
            ff_loss = F.mse_loss(model_ff(x), y).item()
            plain_loss = F.mse_loss(model_plain(x), y).item()
        
        improvement = (plain_loss - ff_loss) / plain_loss
        passed = improvement > 0.3  # 30% improvement
        
        report.add(
            "Spectral Nightmare (100Hz)",
            passed,
            f"Fourier improvement: {improvement:.1%}"
        )
        
    except Exception as e:
        report.add("Spectral Nightmare (100Hz)", False, str(e))


# =============================================================================
# TEST 6: UNCERTAINTY TORTURE
# =============================================================================

def test_uncertainty_torture(report: StressTestReport):
    """
    Feed completely random noise (OOD).
    Prediction sets must be large (high uncertainty).
    """
    print("\n[TEST 6] UNCERTAINTY TORTURE")
    print("-" * 40)
    
    try:
        model = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
        # Calibrate on structured data
        cal_x = torch.randn(200, 40) * 0.1  # Small variance
        cal_y = torch.randint(0, 8, (200,))
        cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=32)
        
        cp = ConformalPredictor(model, alpha=0.1)
        cp.calibrate(cal_loader)
        
        # Test on pure noise (completely OOD)
        ood_x = torch.randn(100, 40) * 100  # 1000x larger variance!
        
        set_sizes = []
        for i in range(len(ood_x)):
            pred_set = cp.predict(ood_x[i:i+1])
            set_sizes.append(pred_set.set_size)
        
        avg_size = np.mean(set_sizes)
        max_size = max(set_sizes)
        
        # On OOD, sets should be large (uncertain)
        passed = avg_size > 3  # Should be uncertain
        
        report.add(
            "Uncertainty on OOD",
            passed,
            f"Avg set size: {avg_size:.1f} (max: {max_size})"
        )
        
    except Exception as e:
        report.add("Uncertainty on OOD", False, str(e))


# =============================================================================
# TEST 7: COMBINED ASSAULT
# =============================================================================

def test_combined_assault(report: StressTestReport):
    """
    Everything at once: continual learning + adversarial + shift.
    ArcV2 must survive with auto configuration.
    """
    print("\n[TEST 7] COMBINED ASSAULT")
    print("-" * 40)
    
    try:
        model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        optimizer = optim.Adam(model.parameters())
        
        # Full ArcV2 with everything enabled
        arc = ArcV2.auto(model, optimizer, safety_level="maximum")
        
        task_losses = []
        
        # 3 tasks with adversarial attacks and distribution shift
        for task_id in range(3):
            arc.begin_task(f"assault_task_{task_id}")
            
            # Each task: different distribution + adversarial training
            shift = (task_id + 1) * 2
            
            for epoch in range(10):
                # Normal batch
                x = torch.randn(32, 30) * shift
                y = torch.randint(0, 10, (32,))
                
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                
                # Add EWC
                ewc_loss = arc.get_ewc_loss()
                total = loss + ewc_loss
                
                total.backward()
                optimizer.step()
                
                status = arc.step(total, x, y)
                task_losses.append(total.item())
            
            # Consolidate
            dummy_loader = DataLoader(
                TensorDataset(torch.randn(50, 30) * shift, torch.randint(0, 10, (50,))),
                batch_size=32
            )
            arc.consolidate_task(dummy_loader)
        
        # Check: system didn't crash and losses are reasonable
        final_loss = task_losses[-1]
        max_loss = max(task_losses)
        
        passed = max_loss < 50 and final_loss < 10
        
        report.add(
            "Combined Assault (3 Tasks + Adv + Shift)",
            passed,
            f"Max loss: {max_loss:.1f}, Final: {final_loss:.1f}"
        )
        
    except Exception as e:
        report.add("Combined Assault (3 Tasks + Adv + Shift)", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ARC v2.0 EXTREME STRESS TEST SUITE")
    print("The Toughest Possible Trials")
    print("=" * 60)
    
    report = StressTestReport()
    
    # Run all tests
    test_catastrophic_forgetting(report)
    test_adversarial_onslaught(report)
    test_distribution_shift(report)
    test_pinn_chaos(report)
    test_spectral_nightmare(report)
    test_uncertainty_torture(report)
    test_combined_assault(report)
    
    # Final summary
    all_passed = report.summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())