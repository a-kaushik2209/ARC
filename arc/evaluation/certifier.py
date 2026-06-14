"""
Self-healing certifier for verifying model recovery from injected failures.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORT THESE FROM FRAMEWORK
from arc.experiments.framework import FailureInducer, FailureType


class SelfHealingCertifier:
    """
    Certifier that verifies self-healing mechanics are working.
    
    Uses FailureInducer from framework.py to inject real failures.
    """
    
    def __init__(self, model, optimizer, training_config, callbacks=None):
        """
        Initialize the certifier.
        
        Args:
            model: The neural network model to test
            optimizer: Optimizer instance (e.g., torch.optim.Adam)
            training_config: Dict with 'lr', 'batch_size', etc.
            callbacks: List of self-healing callback instances
        """
        self.model = model
        self.optimizer = optimizer
        self.config = training_config
        self.callbacks = callbacks or []
        self.test_results = {}
        self.device = next(model.parameters()).device
    
    def run_verification_loop(
        self, 
        failure_type: FailureType,  # CHANGED: use FailureType enum
        steps: int = 30, 
        trigger_step: int = 15,
        severity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run a training loop with failure injection using FailureInducer.
        
        Args:
            failure_type: Type of failure to inject (from FailureType enum)
            steps: Total training steps
            trigger_step: When to inject the failure
            severity: How severe the failure is (1.0 = default)
        
        Returns:
            Dict with test results
        """
        # CREATE THE FAILURE INJECTOR (reusing from framework.py)
        injector = FailureInducer(failure_type=failure_type, severity=severity)
        
        history = {
            'failure_type': failure_type.name,
            'losses': [],
            'gradient_norms': [],
            'recovery_triggered': False,
            'recovery_step': None,
            'passed': False,
            'diagnostic': []
        }
        
        # Create dummy training data
        dummy_input = torch.randn(16, 10, device=self.device)
        dummy_target = torch.randint(0, 2, (16,), device=self.device)
        
        for step in range(steps):
            
            # INJECT FAILURE AT TRIGGER STEP
            if step == trigger_step:
                injector.activate()  # Turn on the inducer
                injector.apply_to_model(self.model)  # Corrupt weights/modes
                injector.apply_to_optimizer(self.optimizer)  # Corrupt learning rate
                history['diagnostic'].append(f"Failure injected at step {step}")
            
            # Forward pass
            output = self.model(dummy_input)
            loss = F.cross_entropy(output, dummy_target)
            
            # Check if loss became NaN/Inf (sign of failure)
            if torch.isnan(loss) or torch.isinf(loss):
                history['recovery_triggered'] = True
                history['diagnostic'].append(f"Anomaly detected at step {step}: loss is {loss.item()}")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Let callbacks try to detect & recover
            for callback in self.callbacks:
                if hasattr(callback, 'on_backward'):
                    callback.on_backward(self.model, loss)
            
            # APPLY GRADIENT CORRUPTION (if active)
            injector.apply_to_gradients(self.model)
            
            # Record gradient norm
            grad_norm = sum(
                p.grad.norm().item() ** 2 
                for p in self.model.parameters() 
                if p.grad is not None
            ) ** 0.5
            history['gradient_norms'].append(grad_norm)
            
            # Step optimizer
            self.optimizer.step()
            
            # Record loss
            loss_val = float(loss.item()) if not torch.isnan(loss) else float('nan')
            history['losses'].append(loss_val)
            
            # Check recovery (loss is normal again after anomaly)
            if history['recovery_triggered'] and not (torch.isnan(loss) or torch.isinf(loss)):
                if history['recovery_step'] is None:
                    history['recovery_step'] = step
                    history['passed'] = True
                    history['diagnostic'].append(f"Recovery detected at step {step}")
        
        return history
    
    def certify(self) -> Dict[str, Any]:
        """
        Run all 4 failure injection tests using FailureInducer.
        
        Tests:
        - VANISHING_GRADIENT
        - EXPLODING_GRADIENT
        - MODE_COLLAPSE
        - DIVERGENCE
        
        Returns:
            JSON-serializable conformance report
        """
        # List of failure types to test
        failure_types_to_test = [
            FailureType.VANISHING_GRADIENT,
            FailureType.EXPLODING_GRADIENT,
            FailureType.MODE_COLLAPSE,
            FailureType.DIVERGENCE,
        ]
        
        for failure_type in failure_types_to_test:
            # Run the test
            result = self.run_verification_loop(
                failure_type=failure_type,
                steps=30,
                trigger_step=15,
                severity=1.0
            )
            
            # Store results
            self.test_results[failure_type.name] = result
        
        # Generate and return the conformance report
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate JSON-serializable conformance report.
        
        Matches the format from ExperimentRunner._save_result().
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_type': self.config.get('model_type', 'unknown'),
                'learning_rate': self.config.get('learning_rate', 0.001),
                'batch_size': self.config.get('batch_size', 16),
            },
            'test_results': {},
            'summary': {}
        }
        
        # Add individual test results
        for failure_name, result in self.test_results.items():
            report['test_results'][failure_name] = {
                'passed': result['passed'],
                'recovery_step': result['recovery_step'],
                'recovery_latency_steps': result['recovery_step'] if result['recovery_step'] else None,
                'losses': result['losses'][:20],  # First 20 losses only
                'gradient_norms': result['gradient_norms'][:20],
                'diagnostic': result['diagnostic']
            }
        
        # Add summary stats
        passed_count = sum(1 for r in self.test_results.values() if r['passed'])
        total_count = len(self.test_results)
        
        report['summary'] = {
            'total_tests': total_count,
            'passed': passed_count,
            'failed': total_count - passed_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0.0,
            'conformance': 'PASS' if passed_count == total_count else 'FAIL'
        }
        
        return report
    
    def save_report(self, filepath: str) -> None:
        """Save report to JSON file."""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to {filepath}")