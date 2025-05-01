from time import time
from typing import Dict, List

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.logger.maybe_load_training_state(self)

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards = []
		ep_lengths = []
		per_ep_rewards = {i: [] for i in range(self.cfg.num_envs)}
		frames = []
		recording = False

		obs = self.eval_env.reset()
		done = torch.tensor([True] * self.cfg.num_envs)
		while len(ep_rewards) < self.cfg.eval_episodes:
			torch.compiler.cudagraph_mark_step_begin()
			action = self.agent.act(obs, t0=torch.tensor(done).cuda(), eval_mode=True)
			obs, reward, done, info = self.eval_env.step(action)
			for env_idx in range(self.cfg.num_envs):
				per_ep_rewards[env_idx].append(reward[env_idx].item())
				# update buffer
				if done[env_idx]:
					ep_rewards.append(sum(per_ep_rewards[env_idx]))
					ep_lengths.append(len(per_ep_rewards[env_idx]))
					per_ep_rewards[env_idx] = []
				# recording related
				if env_idx == 0:
					if recording:
						rendered = self.eval_env.render()  # BHWC
						frames.append(rendered[env_idx])
					elif done[env_idx]:
						recording = False

		if self.cfg.save_video and len(frames) > 0:
			self.logger._wandb.log({
					'videos/eval_video': self.logger._wandb.Video(
						np.stack(frames).transpose(0, 3, 1, 2), fps=15, format='gif'
					),}, step=self._step)

		return dict(
			episode_reward=torch.tensor(ep_rewards).mean().item(),
			episode_length=torch.tensor(ep_lengths).float().mean().item(),
			episode_success=(torch.tensor(ep_rewards) >= 0.0).float().mean().item()
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full(self.env.action_space.shape, float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		seed = np.random.randint(0, 2**32 - 1, size=(self.cfg.num_envs,)).tolist()
		# self.eval_env.reset(seed=seed)
		# seed = np.random.randint(0, 2**32 - 1, size=(self.cfg.num_envs,)).tolist()
		# obs = self.env.reset(seed=seed)
		self.eval_env.reset()
		obs = self.env.reset()

		done = torch.full((self.cfg.num_envs,), True)
		self._tds: Dict[int, List[TensorDict]] = {
      i: [self.to_td(obs[i])]
			for i in range(self.cfg.num_envs)
    }  # holds td/step for each env, over one episode

		while self._step <= self.cfg.steps:
      # Evaluate agent periodically
			# if self._step % self.cfg.eval_freq == 0:
			if abs(self._step % self.cfg.eval_freq) < self.cfg.num_envs:
				self.logger.save_agent(self.agent, identifier=f'step{self._step:09d}')
				self.buffer.save_checkpoint()
				self.logger.save_training_state(self)
				# eval_metrics = self.eval()
				# eval_metrics.update(self.common_metrics())
				# self.logger.log(eval_metrics, 'eval')

			for env_idx in range(self.cfg.num_envs):
				# guard the first and the resume cases
				if done[env_idx] and len(self._tds[env_idx]) > 1:
					# log, add to buffer, and reset
					td = torch.cat(self._tds[env_idx])
					reward = td["reward"].nansum(0).item()  # sum over episode
					train_metrics = dict(
						episode_reward=reward,
						episode_success=1.0 if reward >= 0 else 0.0,  # NOTE: hack to get is_success
						episode_length=len(td),
						episode_terminated=td["terminated"][-1].item(),  # or info["terminated"][env_idx]
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, "train")
					self._ep_idx = self.buffer.add(td)
					self._tds[env_idx] = [self.to_td(obs[env_idx])]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=torch.tensor(done).cuda())
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)

			for env_idx in range(self.cfg.num_envs):
				# 0.29 - final_observation, 1.1 - final_obs
				ob_ = info["final_observation"][env_idx] if done[env_idx] else obs[env_idx]
				self._tds[env_idx].append(
					self.to_td(
						ob_, action[env_idx], reward[env_idx], info["terminated"][env_idx]
					)
				)

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print(f"Pretraining agent on seed data... {num_updates} updates")
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				train_metrics = dict()
				for _ in tqdm(range(num_updates), desc='agent.update', disable=num_updates<128):
					train_metrics.update(self.agent.update(self.buffer))
				train_metrics.update(self.common_metrics())
				self.logger.log(train_metrics, "train")

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)
