import gymnasium as gym


import knotgym  # noqa: F401


import numpy as np


# view wrapper
class Pixels(gym.Wrapper):
  def __init__(self, env, cfg):
    super().__init__(env)
    self.cfg = cfg
    self.env = env
    obs = self.env.observation_space  # 480, 960, 3
    height, combined_width, channels = obs.shape
    width = combined_width // 2
    self.observation_space = gym.spaces.Box(
      low=0, high=255, shape=(channels * 2, height, width), dtype=np.uint8
    )

  def _proc_obs(self, obs):
    # 480, 960, 3 --> 480, 480, 6
    obs = np.concatenate(
      [
        obs[:, : obs.shape[1] // 2, :],
        obs[:, obs.shape[1] // 2 :, :],
      ],
      axis=-1,
    )
    # 480, 480, 6 --> 6, 480, 480
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
  split = "tr"
  logdir = cfg.work_dir / split / f"{rank:04d}"
  task = cfg.task  # tie_unknot
  size = 128
  assert cfg.obs in ("rgb", "state")
  output_pixels = cfg.obs == "rgb"
  env = gym.make(
    "knotgym/Unknot-v0",
    task=task,
    logdir=logdir,
    split=split,
    height=size,
    width=size,
    output_pixels=output_pixels,
    **kwargs,
  )
  if output_pixels:
    env = Pixels(env, cfg)
  if old_api:
    env = OldStepWrapper(env)

  # cfg.discount_max = 0.99  # TODO: temporarily hardcode for these envs, makes comparison to other codebases easier
  # cfg.rho = 0.7  # TODO: increase rho for episodic tasks since termination always happens at the end of a sequence
  return env
