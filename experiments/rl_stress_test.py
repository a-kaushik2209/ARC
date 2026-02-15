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
ARC Phase 13: Reinforcement Learning Stress Test

RL training is notoriously unstable due to:
1. Non-stationary data distribution
2. High variance gradients
3. Catastrophic forgetting of good policies
4. Reward hacking and collapse

This test uses a simple DQN on CartPole to demonstrate ARC's value
in an inherently unstable training domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import json
import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import ArcV2

# Try to import gym, provide fallback
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("gym/gymnasium not installed. Install with: pip install gymnasium")


# =============================================================================
# DQN Network
# =============================================================================

class DQN(nn.Module):
    """Simple DQN for CartPole."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Training Functions
# =============================================================================

def train_dqn(use_arc=False, n_episodes=200, inject_instability=False, 
              instability_type="lr_spike"):
    """Train DQN with optional ARC monitoring."""
    
    if not GYM_AVAILABLE:
        return {"failed": True, "error": "gym not installed"}
    
    env = gym.make("CartPole-v1")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    # ARC setup
    arc = None
    if use_arc:
        arc = ArcV2.auto(policy_net, optimizer, safety_level="paranoid")
    
    buffer = ReplayBuffer()
    
    # Training params
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update = 10
    
    results = {
        "use_arc": use_arc,
        "inject_instability": inject_instability,
        "instability_type": instability_type,
        "episode_rewards": [],
        "losses": [],
        "failed": False,
    }
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        n_steps = 0
        
        # Inject instability at episode 100
        if inject_instability and episode == 100:
            if instability_type == "lr_spike":
                for pg in optimizer.param_groups:
                    pg['lr'] *= 100
                print(f"  [INJECTION] LR spiked to {pg['lr']}")
            elif instability_type == "weight_noise":
                with torch.no_grad():
                    for p in policy_net.parameters():
                        p.add_(torch.randn_like(p) * 10)
                print("  [INJECTION] Added massive weight noise")
        
        done = False
        while not done:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            n_steps += 1
            
            # Training step
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Q-learning update
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * next_q * (1 - dones)
                
                loss = F.mse_loss(current_q.squeeze(), target_q)
                
                # Check for failure
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    print(f"  [FAILURE] Loss exploded: {loss.item()}")
                    results["failed"] = True
                    break
                
                optimizer.zero_grad()
                loss.backward()
                
                # Check gradients
                max_grad = 0
                for p in policy_net.parameters():
                    if p.grad is not None:
                        max_grad = max(max_grad, p.grad.abs().max().item())
                
                if max_grad > 1e6:
                    print(f"  [FAILURE] Gradient explosion: {max_grad}")
                    results["failed"] = True
                    break
                
                optimizer.step()
                
                # ARC monitoring
                if use_arc and arc:
                    arc.step(loss.item())
                
                episode_loss += loss.item()
        
        if results["failed"]:
            break
        
        results["episode_rewards"].append(episode_reward)
        results["losses"].append(episode_loss / max(n_steps, 1))
        
        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Target network update
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Progress
        if episode % 20 == 0:
            avg_reward = np.mean(results["episode_rewards"][-20:])
            print(f"  Episode {episode}: Avg Reward = {avg_reward:.1f}, Epsilon = {epsilon:.3f}")
        
        # Check for policy collapse
        if episode > 50 and np.mean(results["episode_rewards"][-20:]) < 15:
            print(f"  [WARNING] Policy collapse detected")
    
    env.close()
    
    # Final stats
    results["final_avg_reward"] = np.mean(results["episode_rewards"][-20:]) if results["episode_rewards"] else 0
    results["episodes_completed"] = len(results["episode_rewards"])
    
    return results


def run_rl_stress_test():
    """Run RL stress test benchmark."""
    
    print("="*60)
    print("REINFORCEMENT LEARNING STRESS TEST (Phase 13)")
    print("Testing ARC on notoriously unstable RL training")
    print("="*60)
    
    if not GYM_AVAILABLE:
        print("\nCannot run: gymnasium/gym not installed")
        print("   Install with: pip install gymnasium")
        return None
    
    scenarios = [
        {"inject": False, "type": None, "name": "baseline"},
        {"inject": True, "type": "lr_spike", "name": "lr_spike"},
        {"inject": True, "type": "weight_noise", "name": "weight_noise"},
    ]
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name'].upper()}")
        print("="*60)
        
        # Without ARC
        print("\n[1/2] Training WITHOUT ARC...")
        result_no_arc = train_dqn(
            use_arc=False, n_episodes=150,
            inject_instability=scenario["inject"],
            instability_type=scenario["type"]
        )
        status = "FAILED" if result_no_arc["failed"] else f"OK (Reward: {result_no_arc['final_avg_reward']:.1f})"
        print(f"  Result: {status}")
        
        # With ARC
        print("\n[2/2] Training WITH ARC...")
        result_arc = train_dqn(
            use_arc=True, n_episodes=150,
            inject_instability=scenario["inject"],
            instability_type=scenario["type"]
        )
        status = "FAILED" if result_arc["failed"] else f"OK (Reward: {result_arc['final_avg_reward']:.1f})"
        print(f"  Result: {status}")
        
        all_results.append({
            "scenario": scenario["name"],
            "without_arc": result_no_arc,
            "with_arc": result_arc,
            "arc_saved": result_no_arc["failed"] and not result_arc["failed"],
        })
    
    # Summary
    print("\n" + "="*60)
    print("RL STRESS TEST SUMMARY")
    print("="*60)
    
    print("\n| Scenario         | No ARC    | With ARC  | ARC Saved? |")
    print("|------------------|-----------|-----------|------------|")
    
    arc_saves = 0
    for r in all_results:
        no_arc = "FAIL" if r["without_arc"]["failed"] else f"R:{r['without_arc']['final_avg_reward']:.0f}"
        with_arc = "FAIL" if r["with_arc"]["failed"] else f"R:{r['with_arc']['final_avg_reward']:.0f}"
        saved = "YES ✓" if r["arc_saved"] else "No"
        if r["arc_saved"]:
            arc_saves += 1
        print(f"| {r['scenario']:16} | {no_arc:9} | {with_arc:9} | {saved:10} |")
    
    print(f"\nARC saved {arc_saves}/{len(all_results)} failing scenarios")
    
    with open("rl_stress_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to: rl_stress_results.json")
    
    return all_results


if __name__ == "__main__":
    run_rl_stress_test()