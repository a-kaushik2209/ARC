import unittest
import torch
import torch.nn as nn
from arc.evaluation.certifier import SelfHealingCertifier
from arc.experiments.framework import FailureType

class SimpleModel(nn.Module):
    """Simple MLP for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestSelfHealingCertifier(unittest.TestCase):
    
    def setUp(self):
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.training_config = {'lr': 0.001, 'batch_size': 16, 'model_type': 'mlp'}
        self.certifier = SelfHealingCertifier(
            model=self.model,
            optimizer=self.optimizer,
            training_config=self.training_config
        )
    
    def test_certifier_initializes(self):
        """Test that certifier initializes without errors."""
        self.assertIsNotNone(self.certifier)
    
    def test_vanishing_gradient_detection(self):
        """Test that VANISHING_GRADIENT failure is detected."""
        result = self.certifier.run_verification_loop(
            failure_type=FailureType.VANISHING_GRADIENT,  # USE ENUM
            steps=30,
            trigger_step=15
        )
        
        # Check structure
        self.assertIn('losses', result)
        self.assertIn('recovery_triggered', result)
        self.assertIn('recovery_step', result)
    
    def test_exploding_gradient_detection(self):
        """Test that EXPLODING_GRADIENT failure is detected."""
        result = self.certifier.run_verification_loop(
            failure_type=FailureType.EXPLODING_GRADIENT,
            steps=30,
            trigger_step=15
        )
        
        self.assertIsNotNone(result)
    
    def test_full_certification(self):
        """Test that certify() runs all tests and generates report."""
        report = self.certifier.certify()
        
        # Check report structure (matches ExperimentRunner format)
        self.assertIn('timestamp', report)
        self.assertIn('test_results', report)
        self.assertIn('summary', report)
        
        # Check that we ran 4 tests
        self.assertEqual(len(report['test_results']), 4)
        
        # Check summary
        self.assertIn('total_tests', report['summary'])
        self.assertIn('passed', report['summary'])
        self.assertIn('conformance', report['summary'])
    
    def test_save_report(self):
        """Test that report can be saved to JSON."""
        import tempfile
        import json
        
        self.certifier.certify()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        self.certifier.save_report(temp_path)
        
        # Read it back and verify it's valid JSON
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('summary', data)
        
        # Cleanup
        import os
        os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()