import gymnasium
import torch
from gym import spaces
from functools import partial as bind


class Vectorized:
  """
  Vectorized environment for TD-MPC2 online training, backed by gymnasium
  """

  def __init__(self, cfg, env_fn):
    super().__init__()
    self.cfg = cfg

    print(f"Creating {cfg.num_envs} environments...")
    self.env = gymnasium.vector.AsyncVectorEnv(
      [bind(env_fn, cfg, rank=i + 1) for i in range(cfg.num_envs)],
      # autoreset_mode="SameStep",  # https://farama.org/Vector-Autoreset-Mode
    )

    obs_space = self.env.single_observation_space
    act_space = self.env.single_action_space

    self.observation_space = spaces.Box(
      low=obs_space.low,
      high=obs_space.high,
      dtype=obs_space.dtype,
      shape=obs_space.shape,
    )
    self.action_space = spaces.Box(
      low=act_space.low,
      high=act_space.high,
      dtype=act_space.dtype,
      shape=act_space.shape,
    )
    self.max_episode_steps = cfg.episode_length
    self.num_envs = self.env.num_envs

  def rand_act(self):
    return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

  def reset(self, **kwargs):
    # old api
    obs, info = self.env.reset(**kwargs)
    return obs

  def step(self, action):
    # old api
    obs, reward, term, trunc, info = self.env.step(action)
    done = term | trunc
    info["terminated"] = term.astype(float)  # numpy array
    if "success" not in info and "is_success" in info:
      info["success"] = info["is_success"]
    if "success" in info:
      info["success"] = info["success"].astype(float)  # numpy array
    return obs, reward, done, info

  # does not work well with gymnasium == 0.29.1
  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)
