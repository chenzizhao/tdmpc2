import dataclasses
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import hydra
from common import MODEL_SIZE, TASK_SET
from omegaconf import OmegaConf


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def maybe_resume(enabled: bool) -> Tuple[bool, Path, str]:
  """look for slurm id and wandb run id in the canonical logs directory."""
  logs_dir = Path(hydra.utils.get_original_cwd()) / "results" / "baselines" / "tdmpc2"
  slurm_job_id = str(os.getenv("SLURM_JOB_ID", "local"))
  default_work_dir = logs_dir / (
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + f"-{slurm_job_id}"
  )

  if slurm_job_id == "local" or not enabled:
    # not on slurm system
    return False, default_work_dir, None  # start new run
  # look for a directory named after the SLURM job ID
  work_dir = None
  for d in logs_dir.iterdir():
    if d.is_dir() and d.name.endswith(str(slurm_job_id)):
      work_dir = d
      break
  if work_dir is None:
    return False, default_work_dir, None  # start new run
  if not (work_dir / "wandb_run_id.txt").exists():
    return False, work_dir, None
  with open(work_dir / "wandb_run_id.txt", "r") as f:
    wandb_run_id = f.read().strip()
  return True, work_dir, wandb_run_id


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	if cfg.work_dir == "results-baselines-tdmpc2":
		resume, work_dir, wandb_run_id = maybe_resume(enabled=False)
		cfg.work_dir = work_dir
		cfg.wandb_run_id = wandb_run_id
	else:
		cfg.work_dir = (
			Path(hydra.utils.get_original_cwd())
			/ "logs"
			/ cfg.task
			/ str(cfg.seed)
			/ cfg.exp_name
		)
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

	# Multi-task
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		# Account for slight inconsistency in task_dim for the mt30 experiments
		cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])


	# venv eval
	if cfg.eval_episodes > 0 and cfg.eval_episodes < cfg.num_envs:
		cfg.eval_episodes = cfg.num_envs
	return cfg_to_dataclass(cfg)
