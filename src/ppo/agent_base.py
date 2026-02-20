import numpy as np
import torch
import time
from abc import ABC, abstractmethod
from collections import deque
from src.shared.environment_manager import EnvironmentManager
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.agent_template import TemplateAgent
from src.shared.rollout_buffer import RolloutBuffer
from itertools import count


class PPOAgentBase(TemplateAgent, ABC):
    """Agent that serves as a base for PPO algorithm. Contains common methods and properties."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper, model):
        """Initializes the PPO agent with the environment, wandb logger, model, and device.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
            model: The neural network model used by the agent.
        """
        super().__init__(environment, wandb)

        self.actor = model.to(self.device)
        self.memory = RolloutBuffer()

    @abstractmethod
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """Selects an action based on the current state using the model."""
        pass

    @abstractmethod
    def evaluate_actions(
        self, batch_states: torch.Tensor, batch_actions: torch.Tensor
    ) -> tuple:
        """Evaluates actions for a batch of states."""
        pass

    @abstractmethod
    def optimize_model(self, final_state: np.ndarray) -> tuple:
        """Optimizes the model using the collected rollout data."""
        pass

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        last_value: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)
        gae = 0.0
        gamma = self.wdb.get_hyperparameter("gamma")
        lmbda = self.wdb.get_hyperparameter("lambda")
        values = torch.cat([values, last_value], dim=0)

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + gamma * (1 - dones[step]) * values[step + 1]
                - values[step]
            )
            gae = delta + gamma * lmbda * (1 - dones[step]) * gae
            advantages[step] = gae

        # Normalize advantages to stabilize training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def train(self):
        """Main training loop for the PPO agent."""
        episode = 0
        total_steps = 0
        max_steps = self.wdb.get_hyperparameter("total_steps")
        save_interval = self.wdb.get_hyperparameter("save_interval")
        hp_rollout_length = self.wdb.get_hyperparameter("rollout_length")
        best_mean = float("-inf")
        reward_buffer = deque(maxlen=save_interval)
        policy_loss_buffer = deque(maxlen=5)
        value_loss_buffer = deque(maxlen=5)

        while True:
            state = self.env.reset()

            for _ in count(1):
                total_steps += 1

                action, log_prob, value = self.get_action(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                rollout_size = self.memory.add(
                    state, action, log_prob, reward, value, terminated
                )
                state = new_state

                if rollout_size == hp_rollout_length:
                    value_loss, policy_loss = self.optimize_model(state)
                    value_loss_buffer.append(value_loss)
                    policy_loss_buffer.append(policy_loss)
                    self.memory.clear()
                    self.wdb.log(
                        {
                            "Total Steps": total_steps,
                            "Value Loss": np.mean(value_loss_buffer),
                            "Policy Loss": np.mean(policy_loss_buffer),
                        }
                    )

                if terminated or truncated:
                    episode_steps, episode_reward = self.env.get_episode_info()
                    episode += 1

                    reward_buffer.append(episode_reward)
                    mean = np.sum(reward_buffer) / save_interval

                    if mean > best_mean and episode > save_interval:
                        best_mean = mean
                        self.save_model(self.actor)
                        print(
                            f"Episode {episode} -- saving model with new best mean reward: {mean}"
                        )

                    if "success_rate" in info and episode > save_interval:
                        self.wdb.log({"Success Rate": info["success_rate"]})

                    if episode % 10 == 0:
                        print(
                            f"Episode {episode} finished in {episode_steps} steps with reward {episode_reward}. "
                            f"Total steps: {total_steps}/{max_steps}."
                        )
                    break

            episode_steps, episode_reward = self.env.get_episode_info()
            self.wdb.log(
                {
                    "Total Steps": total_steps,
                    "Episode Length": episode_steps,
                    "Episode Reward": episode_reward,
                    "Rolling Return": np.mean(reward_buffer),
                }
            )

            if total_steps > max_steps:
                print("The training has successfully finished!")
                break

        self.save_artifact()
        self.wdb.finish()

    def play(self, delay: bool = False) -> tuple:
        """See the agent perform in selected environment."""
        state = self.env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _, _ = self.get_action(state, deterministic=True)
            state, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if delay:
                time.sleep(0.5)

        steps, reward = self.env.get_episode_info()
        return reward, steps, info
