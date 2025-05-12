import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
import warnings

warnings.filterwarnings("ignore")

import json
import re
import sys
from datetime import datetime

import hydra
import numpy as np
import torch
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from termcolor import colored

from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def parse_slurm_job_id(load_from) -> str:
  # given: results/baselines/tdmpc2/2025-05-05-09-07-55-102895-7858394/models/step001000000.pt
  # 7858394
  match = re.search(
    r"results/baselines/tdmpc2/\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}-(\d+)/",
    load_from,
  )
  return match.group(1)


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

  ep_rewards = []
  ep_lengths = []
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

  env.close()

  report = dict(
    # checkpointing
    eval_slurm_job_id=parse_slurm_job_id(cfg.checkpoint),
    load_from=cfg.checkpoint,
    command=" ".join(sys.argv),
    # tasks
    task=cfg.task,
    num_envs=cfg.num_envs,
    task_max_n_crossings=cfg.task_max_n_crossings,
    task_max_n_states=cfg.task_max_n_states,
    # results
    episode_success_rate=np.mean([v >= 0 for v in ep_rewards]).item(),
    episode_rewards_mean=np.mean(ep_rewards).item(),
    episode_lengths_mean=np.mean(ep_lengths, dtype=float).item(),
    n_eval_episodes=len(ep_rewards),
    episode_rewards=ep_rewards,
    episode_lengths=ep_lengths,
    # aux
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    seed=cfg.seed
  )
  report_path = os.path.join(cfg.work_dir, "eval_report.json")
  with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"Eval report saved saved to:{report_path}")


if __name__ == "__main__":
  evaluate()
