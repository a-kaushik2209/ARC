import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class TestSelfHealingArc:
    """Tests for SelfHealingArc."""
    
    def test_baseline_clipping_flag(self):
        """Test that action.gradients_clipped is set correctly when baseline clipping fires."""
        model = nn.Sequential(
            nn.Linear(10, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Configure to ensure baseline clipping fires
        config = SelfHealingConfig(
            enable_baseline_clipping=True,
            gradient_clip_norm=1.0,
            gradient_explosion_threshold=1e9,  # Prevent explosion branch
            enable_forecasting=False           # Prevent forecasting branch
        )
        
        arc = SelfHealingArc(model, optimizer, config=config)
        
        # Forward pass to create gradients
        x = torch.randn(2, 10)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # Call post_backward which should trigger baseline clipping
        action = arc.post_backward()
        
        # Verify the flag was set
        assert action.gradients_clipped is True, "Baseline clipping should set gradients_clipped to True"

    def test_baseline_clipping_opt_out(self):
        """Test that baseline clipping can be opted out using the config flag."""
        model = nn.Sequential(
            nn.Linear(10, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Configure to disable baseline clipping
        config = SelfHealingConfig(
            enable_baseline_clipping=False,
            gradient_clip_norm=1.0,
            gradient_explosion_threshold=1e9,
            enable_forecasting=False
        )
        
        arc = SelfHealingArc(model, optimizer, config=config)
        
        # Forward pass to create gradients
        x = torch.randn(2, 10)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # Call post_backward
        action = arc.post_backward()
        
        # Verify the flag was NOT set
        assert action.gradients_clipped is False, "Baseline clipping should not run when disabled"
