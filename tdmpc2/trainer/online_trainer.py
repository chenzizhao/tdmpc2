from time import time
from typing import Dict, List

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
		for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
			obs, done, ep_reward, t = self.env.reset(), torch.tensor(False), 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done.any():
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=torch.cat(ep_rewards).mean(),
			episode_success=info['success'].mean(),  # TODO: fix vecenv + episodic
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
		train_metrics = {}
		obs = self.env.reset()
		done = torch.full((self.cfg.num_envs,), True)
		eval_next = True
		self._tds: Dict[int, List[TensorDict]] = {
      i: [self.to_td(obs[i])]
			for i in range(self.cfg.num_envs)
    }  # holds td/step for each env, over one episode

		while self._step <= self.cfg.steps:
      # Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

      # Reset environment  # now handled by autoreset in async vec env
			if done.any():
				if eval_next:
					print("eval during training is disabled for now")
	      # TODO: bring back eval()
        # 	eval_metrics = self.eval()
        # 	eval_metrics.update(self.common_metrics())
        # 	self.logger.log(eval_metrics, 'eval')
					self.logger.save_agent(self.agent, identifier=f'step{self._step:09d}')
					eval_next = False

			if self._step > 0:
				for env_idx in range(self.cfg.num_envs):
					if done[env_idx]:  # log, add to buffer, and reset
						td = torch.cat(self._tds[env_idx])
						reward = td["reward"].nansum(0).item()  # sum over episode
						train_metrics.update(
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

			# TODO: is this slow? adding to buffer one env by one
			for env_idx in range(self.cfg.num_envs):
				ob_ = info["final_obs"][env_idx] if done[env_idx] else obs[env_idx]
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
				_train_metrics = dict()
				for _ in tqdm(range(num_updates), desc='agent.update', disable=num_updates<128):
					_train_metrics.update(self.agent.update(self.buffer))
				train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)
