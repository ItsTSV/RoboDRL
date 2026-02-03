import gymnasium as gym
import shimmy
import dm_control
from gymnasium import wrappers
import numpy as np


class EnvironmentManager:
    """Provides a manager that builds gymnasium environment and handles
    interaction and data processing."""

    def __init__(self, name: str, render_mode: str):
        """Initializes Gymnasium environment and info about it

        Attributes:
            name (str): Selected environment, see https://gymnasium.farama.org/
            render_mode (str): "rgb_array" for computations, "human" for showcases
        """
        self.env = gym.make(name, render_mode=render_mode)
        self.episode_steps = 0
        self.episode_reward = 0
        self.observation_norm_wrapper = None
        self.normalize_rewards = False

    def build_continuous(self):
        """Wraps itself in wrappers that are used for environments with continuous action space.

        Flatten Observation -- flattens dict observations into a single array.
        Clip actions -- normalises the input action to [-1, 1] range.
        Normalize Observation -- normalises observations to have mean 0 and variance 1.
        """
        if isinstance(self.env.observation_space.sample(), dict):
            self.env = wrappers.FlattenObservation(self.env)
        self.env = wrappers.ClipAction(self.env)
        self.env = wrappers.NormalizeObservation(self.env)
        self.observation_norm_wrapper = self.env

    def build_reward_normalization(self):
        """Wraps itself in wrappers that normalizes rewards and records episode statistics.

        RecordEpisodeStatistics -- records episode length and total reward in 'info' dict.
        NormalizeReward -- normalizes rewards to have mean 0 and variance 1.
        """
        self.env = wrappers.RecordEpisodeStatistics(self.env)
        self.env = wrappers.NormalizeReward(self.env)
        self.normalize_rewards = True

    def build_video_recorder(self, video_folder: str = "outputs/", fps: int = 60):
        """Wraps itself in a video recorder wrapper."""
        self.env.metadata["render_fps"] = fps
        self.env = wrappers.RecordVideo(
            self.env,
            video_folder,
            episode_trigger=lambda episode_id: True,
            name_prefix="agent_video"
        )

    def save_normalization_parameters(self, path):
        """Saves the observation normalization mean and variance to a file."""
        if self.observation_norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        rms = self.observation_norm_wrapper.obs_rms
        np.savez(path, mean=rms.mean, var=rms.var, count=rms.count)

    def load_normalization_parameters(self, path):
        """Loads the observation normalization mean and variance from a file."""
        if self.observation_norm_wrapper is None:
            raise ValueError(
                "Normalization wrapper not found. Ensure build_continuous() has been called."
            )
        print("Loaded!")
        data = np.load(path)
        rms = self.observation_norm_wrapper.obs_rms
        rms.mean = data["mean"]
        rms.var = data["var"]
        rms.count = data["count"]

    def get_dimensions(self) -> tuple:
        return (
            (
                len(self.env.action_space.sample())
                if isinstance(self.env.action_space, gym.spaces.Box)
                else self.env.action_space.n
            ),
            len(self.env.observation_space.sample()),
        )

    def get_action_bounds(self) -> tuple:
        return self.env.action_space.low, self.env.action_space.high

    def get_state_shape(self) -> tuple:
        return self.env.observation_space.shape

    def get_random_action(self) -> np.ndarray:
        return self.env.action_space.sample()

    def step(self, action) -> tuple:
        """Advances the environment, processes the output

        Returns:
            New observation, reward acquired by performing action,
            termination info, truncation info, additional env data
        """
        state, reward, terminated, truncated, info = self.env.step(action)

        self.episode_steps += 1
        if self.normalize_rewards:
            if "episode" in info:
                self.episode_reward = info["episode"]["r"],
        else:
            self.episode_reward += reward

        return state, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        """Resets the environment, returns a new state"""
        new_state = self.env.reset()[0]
        self.episode_steps = 0
        self.episode_reward = 0
        return new_state

    def get_episode_info(self) -> tuple:
        return self.episode_steps, self.episode_reward

    def close(self) -> None:
        self.env.close()

    def render(self):
        return self.env.render()
