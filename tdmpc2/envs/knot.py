import gymnasium as gym


import knotgym  # noqa: F401


import numpy as np


# view wrapper
class Pixels(gym.Wrapper):
  def __init__(self, env, cfg):
    super().__init__(env)
    self.cfg = cfg
    self.env = env
    height, width, channels = self.env.observation_space.shape  # 480, 960, 3
    self.observation_space = gym.spaces.Box(
      low=0, high=255, shape=(channels, height, width), dtype=np.uint8
    )

  def _proc_obs(self, obs):
    # 480, 960, 3 --> 3, 480, 960
    obs = np.transpose(obs, (2, 0, 1))
    return obs

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    return self._proc_obs(obs), info

  def step(self, action):
    obs, reward, term, trunc, info = self.env.step(action)
    return self._proc_obs(obs), reward, term, trunc, info


class OldStepWrapper(gym.Wrapper):
  def step(self, action):
    obs, reward, term, trunc, info = self.env.step(action)
    done = trunc or term
    info["terminated"] = term
    info["success"] = info["is_success"]
    return obs, reward, done, info

  def reset(self, **kwargs):
    obs, _ = self.env.reset(**kwargs)
    return obs


def make_env(cfg, rank=1, old_api=True, **kwargs):
  split = kwargs.pop("split")
  logdir = cfg.work_dir / split / f"{rank:04d}" if rank == 1 else None
  task = cfg.task  # tie_unknot
  r_gc_allow_flipped_or_mirrored = cfg.get(
    "r_gc_allow_flipped_or_mirrored", False
  )
  size = 128
  assert cfg.obs in ("rgb", "state")
  output_pixels = cfg.obs == "rgb"
  if cfg.episode_length != "???":
    kwargs["duration"] = cfg.episode_length
  if cfg.task_max_n_states is not None:
    kwargs["task_max_n_states"] = cfg.task_max_n_states
  if cfg.task_max_n_crossings is not None:
    kwargs["task_max_n_crossings"] = cfg.task_max_n_crossings
  kwargs["task_subset_seed"] = cfg.task_subset_seed
  kwargs["render_both"] = cfg.render_both
  kwargs["reset_noise_scale"] = cfg.reset_noise_scale
  env = gym.make(
    "knotgym/Unknot-v0",
    task=task,
    logdir=logdir,
    logfreq=100,  #
    split=split,
    height=size,
    width=size,
    output_pixels=output_pixels,
    r_gc_allow_flipped_or_mirrored=r_gc_allow_flipped_or_mirrored,
    **kwargs,
  )
  if output_pixels:
    env = Pixels(env, cfg)
  if old_api:
    env = OldStepWrapper(env)

  # cfg.discount_max = 0.99  # TODO: temporarily hardcode for these envs, makes comparison to other codebases easier
  # cfg.rho = 0.7  # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
  return env
