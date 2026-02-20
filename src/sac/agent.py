import numpy as np
import torch
import time
import itertools
import torch.nn.functional as F
from collections import deque
from src.shared.environment_manager import EnvironmentManager
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.agent_template import TemplateAgent
from .models import QNet, ActorNet
from src.shared.replay_buffer import ReplayBuffer
from itertools import count


class SACAgent(TemplateAgent):
    """Agent that implements Soft Actor Critic algorithm."""

    def __init__(self, environment: EnvironmentManager, wandb: WandbWrapper):
        """Initializes the SAC agent with environment, wandb, models, and memory.

        Args:
            environment (EnvironmentManager): The environment in which the agent operates.
            wandb (WandbWrapper): Wandb wrapper for tracking and hyperparameter management.
        """
        super().__init__(environment, wandb)

        network_size = self.wdb.get_hyperparameter("network_size")
        action_count, state_count = self.env.get_dimensions()

        self.actor = ActorNet(action_count, state_count, network_size).to(self.device)

        self.qnet1 = QNet(action_count, state_count, network_size).to(self.device)
        self.qnet2 = QNet(action_count, state_count, network_size).to(self.device)

        self.qnet1_target = QNet(action_count, state_count, network_size).to(
            self.device
        )
        self.qnet2_target = QNet(action_count, state_count, network_size).to(
            self.device
        )

        self.target_entropy = -float(action_count)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.qnet1_target.load_state_dict(self.qnet1.state_dict())
        self.qnet2_target.load_state_dict(self.qnet2.state_dict())

        memory_size = self.wdb.get_hyperparameter("memory_size")
        self.memory = ReplayBuffer(memory_size, action_count, state_count)

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.wdb.get_hyperparameter("learning_rate_actor"),
        )
        self.optimizer_q = torch.optim.AdamW(
            itertools.chain(self.qnet1.parameters(), self.qnet2.parameters()),
            lr=self.wdb.get_hyperparameter("learning_rate_q"),
            weight_decay=1e-5
        )
        self.optimizer_alpha = torch.optim.Adam(
            [self.log_alpha], lr=self.wdb.get_hyperparameter("learning_rate_actor")
        )

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> tuple:
        """Selects an action based on the current state using the actor.

        Returns:
            action: Action tensor with [-1, 1 bounds]
            log_prob: Log probability of selected action
        """
        log_std_min = self.wdb.get_hyperparameter("log_std_min")
        log_std_max = self.wdb.get_hyperparameter("log_std_max")

        mean, log_std = self.actor(state)

        if deterministic:
            action = torch.tanh(mean)
            return action, None

        # Normalise (OpenAI version)
        log_std = torch.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        # Distribution and re-parametrisation trick
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Because tanh was used, logprobs need to be adjusted
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def optimize_q_networks(self) -> tuple:
        """Optimizes Q networks using data from memory

        Returns:
            tuple (float, float. float): Total Q-Loss, Q1 loss, Q2 loss
        """
        # Sample data from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        dones = dones.float()

        # Compute next Q-values using Q-target networks
        gamma = self.wdb.get_hyperparameter("gamma")
        alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action(next_states)
            next_targets_qnet1 = self.qnet1_target(next_states, next_actions)
            next_targets_qnet2 = self.qnet2_target(next_states, next_actions)
            min_next_targets = torch.min(next_targets_qnet1, next_targets_qnet2)
            next_q_values = rewards + gamma * (1 - dones) * (
                min_next_targets - alpha * next_log_probs
            )

        # Get current Q-values using Q-policy networks
        q1_values = self.qnet1(states, actions)
        q2_values = self.qnet2(states, actions)

        # Compute losses
        q1_loss = F.mse_loss(q1_values, next_q_values)
        q2_loss = F.mse_loss(q2_values, next_q_values)
        q_loss = q1_loss + q2_loss

        # Optimise Q-policy networks via gradient descent
        self.optimizer_q.zero_grad()
        q_loss.backward()

        # Clip gradients
        max_grad_norm = self.wdb.get_hyperparameter("max_grad_norm")
        torch.nn.utils.clip_grad_norm_(self.qnet1.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.qnet2.parameters(), max_grad_norm)
        self.optimizer_q.step()

        return q_loss.item(), q1_loss.item(), q2_loss.item()

    def optimize_actor_network(self) -> tuple:
        """Optimizes actor network and entropy temperature using data from memory"""
        # Sample data from memory
        batch_size = self.wdb.get_hyperparameter("batch_size")
        states, _, _, _, _ = self.memory.sample(batch_size)

        # Get current_actions and their Q-values
        current_actions, log_probs = self.get_action(states)
        q1_values = self.qnet1(states, current_actions)
        q2_values = self.qnet2(states, current_actions)
        min_q_values = torch.min(q1_values, q2_values)

        # Compute actor loss
        alpha = self.log_alpha.exp().item()
        actor_loss = log_probs * alpha - min_q_values
        actor_loss = actor_loss.mean()

        # Optimize actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()

        # Clip gradients
        max_grad_norm = self.wdb.get_hyperparameter("max_grad_norm")
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.optimizer_actor.step()

        # Alpha loss
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        # Optimize alpha
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_grad_norm)
        self.optimizer_alpha.step()

        return actor_loss.item(), alpha_loss.item(), alpha

    def polyak_update(self, source: torch.nn.Module, target: torch.nn.Module):
        """Updates target networks by polyak averaging."""
        tau = self.wdb.get_hyperparameter("tau")
        with torch.no_grad():
            for src_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    tau * src_param.data + (1 - tau) * target_param.data
                )

    def train(self):
        """Main training loop for the SAC agent."""
        episode = 0
        total_steps = 0
        max_steps = self.wdb.get_hyperparameter("total_steps")
        warmup_steps = self.wdb.get_hyperparameter("warmup_steps")
        save_interval = self.wdb.get_hyperparameter("save_interval")
        policy_update_frequency = self.wdb.get_hyperparameter("policy_update_frequency")
        reward_buffer = deque(maxlen=save_interval)
        best_mean = float("-inf")

        while True:
            state = self.env.reset()

            for _ in count(1):
                total_steps += 1

                if total_steps < warmup_steps:
                    action = self.env.get_random_action()
                else:
                    state_tensor = torch.tensor(state).to(self.device).unsqueeze(0)
                    action, log_probs = self.get_action(state_tensor)
                    action = action.detach().cpu().numpy()[0]

                next_state, reward, terminated, truncated, info = self.env.step(action)
                scaled_reward = reward * self.wdb.get_hyperparameter("reward_scale")

                self.memory.add(state, action, scaled_reward, next_state, terminated)
                state = next_state

                if total_steps > warmup_steps:
                    q_loss, q1_loss, q2_loss = self.optimize_q_networks()
                    self.polyak_update(self.qnet1, self.qnet1_target)
                    self.polyak_update(self.qnet2, self.qnet2_target)

                    # Burst update -> "Hack" to improve performance
                    if total_steps % policy_update_frequency == 0:
                        for _ in range(policy_update_frequency):
                            actor_loss, alpha_loss, alpha = self.optimize_actor_network()

                    if total_steps % 100 == 0:
                        self.wdb.log(
                            {
                                "Total Steps": total_steps,
                                "Q Loss": q_loss,
                                "Q1 Loss": q1_loss,
                                "Q2 Loss": q2_loss,
                                "Actor Loss": actor_loss,
                                "Alpha Loss": alpha_loss,
                                "Alpha": alpha,
                            }
                        )

                if terminated or truncated:
                    episode_steps, episode_reward = self.env.get_episode_info()
                    episode += 1

                    reward_buffer.append(episode_reward)
                    mean = np.mean(reward_buffer)

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

    def play(self, delay: bool = False):
        """See the agent perform in selected environment."""
        state = self.env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            state = torch.tensor(state).to(self.device).unsqueeze(0)
            action, _ = self.get_action(state, deterministic=True)
            action = action.detach().cpu().numpy()[0]
            state, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if delay:
                time.sleep(0.5)

        steps, reward = self.env.get_episode_info()
        return reward, steps, info
