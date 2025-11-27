"""
test_arpos.py
-------------
Unit tests for ARPOS reinforcement learning components.

These tests validate:
- Environment reset/step behavior
- Agent action selection
- Training loop integration
"""

import torch
from src.arpos.envs import RehabEnv
from src.arpos.agents import PPOAgent
from src.arpos.optimize import train_agent


def test_rehab_env_step():
    """
    Ensure the RehabEnv environment behaves correctly.
    """
    env = RehabEnv()
    state = env.reset()

    # State dimension should match environment definition
    assert isinstance(state, (list, tuple, torch.Tensor)), "State must be tensor/list"
    assert len(state) == env.state_dim, "State dimension mismatch"

    action = env.sample_action()
    next_state, reward, done, info = env.step(action)

    assert len(next_state) == env.state_dim
    assert isinstance(reward, (float, int)), "Reward must be numeric"
    assert isinstance(done, bool), "Done flag must be boolean"


def test_ppo_agent_action():
    """
    Test PPO agent action selection.
    """
    env = RehabEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)

    state = env.reset()
    action = agent.select_action(state)

    assert 0 <= action < env.action_dim, "Action out of bounds"


def test_arpos_training_short_run():
    """
    Run tiny ARPOS training loop to verify no runtime errors.
    """
    env = RehabEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)

    # Short training for test speed
    train_agent(agent, env, max_episodes=3, max_steps=10)
