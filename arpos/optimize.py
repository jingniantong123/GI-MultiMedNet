"""
optimize.py
-----------
Optimization pipeline for ARPOS (Adaptive Rehabilitation and Performance Optimization Strategy).

This file implements:
- Training loop for RL agents (PPO, DQN, etc.)
- Evaluation utilities
- Checkpoint saving and loading
- Integration with custom rehabilitation environments

This module assumes:
    src/arpos/
        agents/   -> contains agent classes (e.g., PPOAgent, DQNAgent)
        envs/     -> contains Env classes (e.g., RehabEnv)
        optimize.py (this file)

Author: Your Name
"""

import os
import time
import torch
import numpy as np

from agents import PPOAgent, DQNAgent
from envs import RehabEnv


# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------
def train_agent(
    agent,
    env,
    max_episodes=300,
    max_steps=200,
    eval_interval=20,
    save_path="arpos_checkpoint.pth",
):
    """
    Training loop for ARPOS.

    Args:
        agent: RL agent (from agents/)
        env: Rehab environment
        max_episodes: number of training episodes
        max_steps: max interaction steps per episode
        eval_interval: run evaluation every N episodes
        save_path: checkpoint file

    Returns:
        None
    """

    reward_history = []

    print("===== ARPOS Training Started =====")
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        reward_history.append(episode_reward)

        print(f"Episode {episode}/{max_episodes} | Reward: {episode_reward:.2f}")

        # Periodic evaluation
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env)
            print(f"[Evaluation] Avg Reward: {eval_reward:.2f}")

            # Save checkpoint
            save_checkpoint(agent, save_path)
            print(f"Checkpoint saved to {save_path}")

    print("===== ARPOS Training Complete =====")


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate_agent(agent, env, episodes=5, max_steps=200):
    """
    Evaluate agent performance without learning.

    Returns average episode reward.
    """
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state, explore=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

        total_reward += episode_reward

    return total_reward / episodes


# -----------------------------------------------------------
# Checkpoint Saving / Loading
# -----------------------------------------------------------
def save_checkpoint(agent, save_path: str):
    """
    Saves agent model weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "optimizer": agent.optimizer.state_dict()
        },
        save_path
    )


def load_checkpoint(agent, checkpoint_path: str):
    """
    Loads model weights into an agent.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    data = torch.load(checkpoint_path)
    agent.policy.load_state_dict(data["policy"])
    agent.optimizer.load_state_dict(data["optimizer"])

    print(f"Loaded checkpoint from {checkpoint_path}")


# -----------------------------------------------------------
# Standalone Training Runner
# -----------------------------------------------------------
def run_training(
    algo="ppo",
    save_path="checkpoints/arpos_agent.pth",
    episodes=300,
):
    """
    Entry point for ARPOS RL training.

    Args:
        algo: "ppo" or "dqn"
        save_path: where to save model weights
    """

    print("Initializing ARPOS Environment...")
    env = RehabEnv()

    print(f"Selected RL Algorithm: {algo.upper()}")

    if algo.lower() == "ppo":
        agent = PPOAgent(env.state_dim, env.action_dim)
    elif algo.lower() == "dqn":
        agent = DQNAgent(env.state_dim, env.action_dim)
    else:
        raise ValueError("Unsupported algorithm. Choose 'ppo' or 'dqn'.")

    train_agent(
        agent=agent,
        env=env,
        max_episodes=episodes,
        save_path=save_path,
    )


# -----------------------------------------------------------
# CLI Support
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ARPOS RL Optimization Runner"
    )

    parser.add_argument("--algo", type=str, default="ppo",
                        help="RL algorithm: ppo or dqn")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes")
    parser.add_argument("--save", type=str, default="checkpoints/arpos_agent.pth",
                        help="Path to save agent checkpoint")

    args = parser.parse_args()

    run_training(
        algo=args.algo,
        save_path=args.save,
        episodes=args.episodes
    )
