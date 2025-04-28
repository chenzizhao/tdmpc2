import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
import json

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
  """
  Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

  Most relevant args:
          `task`: task name (or mt30/mt80 for multi-task evaluation)
          `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
          `checkpoint`: path to model checkpoint to load
          `eval_episodes`: number of episodes to evaluate on per task (default: 10)
          `save_video`: whether to save a video of the evaluation (default: True)
          `seed`: random seed (default: 1)

  See config.yaml for a full list of args.

  Example usage:
  ````
          $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
          $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
          $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
  ```
  """
  assert torch.cuda.is_available()
  assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
  cfg = parse_cfg(cfg)
  set_seed(cfg.seed)
  print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
  print(
    colored(
      f"Model size: {cfg.get('model_size', 'default')}", "blue", attrs=["bold"]
    )
  )
  print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

  # Make environment
  env = make_env(cfg, eval_mode=True)

  # Load agent
  agent = TDMPC2(cfg)
  assert os.path.exists(cfg.checkpoint), (
    f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
  )
  agent.load(cfg.checkpoint)

  print(colored(f"Evaluating agent on {cfg.task}:", "yellow", attrs=["bold"]))
  if cfg.save_video:
    video_dir = os.path.join(cfg.work_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

  ep_rewards = []
  ep_lengths = []
  frames = []
  should_record = True
  obs = env.reset()
  done = torch.tensor([True] * env.num_envs)
  per_env_rewards = {i: [] for i in range(env.num_envs)}  # buffer for venv
  while len(ep_rewards) < cfg.eval_episodes:
    action = agent.act(obs, t0=torch.tensor(done).cuda(), eval_mode=True)
    obs, reward, done, info = env.step(action)

    for env_idx in range(env.num_envs):
      per_env_rewards[env_idx].append(reward[env_idx].item())
      if done[env_idx]:
        ep_rewards.append(sum(per_env_rewards[env_idx]))
        ep_lengths.append(len(per_env_rewards[env_idx]))
        per_env_rewards[env_idx] = []
      if env_idx == 0 and cfg.save_video and should_record:
        frames.append(env.render())

  if cfg.save_video:
    imageio.mimsave(os.path.join(video_dir, "eval.gif"), frames, fps=15)

  ep_rewards = np.mean(ep_rewards)
  ep_successes = np.mean(np.array(ep_rewards) >= 0.0)
  ep_lengths = np.mean(ep_lengths, dtype=np.float32)

  print(colored("Evaluation results:", "green", attrs=["bold"]))
  print(colored(f"  Average episode reward: {ep_rewards:.2f}", "green"))
  print(colored(f"  Average episode length: {ep_lengths:.2f}", "green"))
  print(colored(f"  Average episode success rate: {ep_successes:.2f}", "green"))

  # add timestamps

  report = {
    "episode_reward": ep_rewards,
    "episode_length": ep_lengths,
    "episode_success": ep_successes,
  }
  with open(os.path.join(cfg.work_dir, "eval_results.txt"), "w") as f:
    json.dump(report, f, indent=4)


if __name__ == "__main__":
  evaluate()
