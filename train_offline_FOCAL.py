# offline meta rl with contrastive representation learning 
# encoder and agent training are disentangled
# FOCAL: sample pos/neg pairs from same/diff task replay buffers
# relabel-gt: sample neg pairs with gt reward/state relabelling
# relabel-separate: learn reward/transition models for each task, sample neg pairs with the learned relabelling models
# ours: learn conditional generative model over all tasks, sample neg pairs with the learned generative model

import os
import sys
import time
import argparse
import torch
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_cheetah_vel, args_cheetah_dir, args_ant_dir, args_hopper_param, args_walker_param
import numpy as np

from models.encoder import RNNEncoder, MLPEncoder
from models.decoder import FOCALDecoder
from algorithms.dqn import DQN
from algorithms.sac import SAC
from algorithms.focalsac import FOCALSAC
from environments.make_env import make_env
import torchkit.pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy
from torch.optim import Adam
from torchkit.brac import divergences
from utils.visual_offline_dataset import visual_by_tsne
import copy
import mbrl


class FOCAL:
	# algorithm class of offline meta-rl with contrastive learning
	# training: (learn models), sample pos/neg pairs (with relabelling), train encoder, train dqn/sac
	# testing: given task context set, extract task encoding, rollout policy in the env

	def __init__(self, args, train_dataset, train_goals, eval_dataset, eval_goals):
		"""
		Seeds everything.
		Initialises: logger, environments, policy (+storage +optimiser).
		"""

		self.args = args

		# make sure everything has the same seed
		utl.seed(self.args.seed)

		# initialize tensorboard logger 
		if self.args.log_tensorboard:
			self.tb_logger = TBLogger(self.args)

		self.args, _ = off_utl.expand_args(self.args, include_act_space=True)
		if self.args.act_space.__class__.__name__ == "Discrete":
			self.args.policy = 'dqn'
		else:
			self.args.policy = 'sac'

		# load augmented buffer to self.storage 
		self.load_buffer(train_dataset, train_goals) 
		
		
		if self.args.pearl_deterministic_encoder:  
			self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size
		else:
			self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size * 2
		self.goals = train_goals
		self.eval_goals = eval_goals
		# context set, to extract task encoding
		self.context_dataset = train_dataset       
		self.eval_context_dataset = eval_dataset


		# initialize policy
		self.initialize_policy()
		# initialize task encoder
		if args.encoder_type == 'rnn':
			self.encoder = RNNEncoder(
				layers_before_gru=self.args.layers_before_aggregator,
				hidden_size=self.args.aggregator_hidden_size,
				layers_after_gru=self.args.layers_after_aggregator,
				task_embedding_size=self.args.task_embedding_size,
				action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim, # fixed a bug?
				action_embed_size=self.args.action_embedding_size,
				state_size=self.args.obs_dim,
				state_embed_size=self.args.state_embedding_size,
				reward_size=1,
				reward_embed_size=self.args.reward_embedding_size,
			).to(ptu.device)
		elif args.encoder_type == 'mlp':
			self.encoder = MLPEncoder(
					hidden_size=self.args.aggregator_hidden_size,
					num_hidden_layers=2,
					task_embedding_size=self.args.task_embedding_size,
					action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim,
					state_size=self.args.obs_dim,
					reward_size=1,
					term_size=1,
					stochasity=self.args.enc_stochastic
				).to(ptu.device)
		else:
			raise NotImplementedError
		
		self.decoder = FOCALDecoder(
			obs_size=self.args.obs_dim,
			action_size=self.args.action_dim,
			task_embedding_size=self.args.task_embedding_size,
			task_embedd_is_deterministic=False,
			device=ptu.device,
			num_layers=2,
			hidden_size=self.args.aggregator_hidden_size,
			ensemble_size=self.args.ensemble_size,
		).to(ptu.device)
		
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr)
		self.finetune_encoder_decoder_optimizer = torch.optim.Adam([{"params": self.encoder.parameters()}, {"params": self.decoder.parameters()}], lr=self.args.encoder_lr)

		# load vae for inference in evaluation
		#self.load_vae()

		# create environment for evaluation    
		self.env = make_env(args.env_name,
							args.max_rollouts_per_task,
							seed=args.seed,
							n_tasks=self.args.num_eval_tasks)
		# fix the possible eval goals to be the testing set's goals
		self.env.set_all_goals(eval_goals)
		self.env_train = make_env(args.env_name,
							args.max_rollouts_per_task,
							seed=args.seed,
							n_tasks=self.args.num_train_tasks) 
		self.env_train.set_all_goals(train_goals)

		#if self.args.env_name == 'GridNavi-v2' or self.args.env_name == 'GridBlock-v2':
		#    self.env.unwrapped.goals = [tuple(goal.astype(int)) for goal in self.goals]
		#print(self.evaluate())
		#sys.exit(0)


	def initialize_policy(self):
		if self.args.policy == 'dqn':
			q_network = FlattenMlp(input_size=self.args.augmented_obs_dim,
								   output_size=self.args.act_space.n,
								   hidden_sizes=self.args.dqn_layers)
			self.agent = DQN(
				q_network,
				# optimiser_vae=self.optimizer_vae,
				lr=self.args.policy_lr,
				gamma=self.args.gamma,
				tau=self.args.soft_target_tau,
			).to(ptu.device)
		else:
			# assert self.args.act_space.__class__.__name__ == "Box", (
			#     "Can't train SAC with discrete action space!")
			q1_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
									output_size=1,
									hidden_sizes=self.args.dqn_layers)
			q2_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
									output_size=1,
									hidden_sizes=self.args.dqn_layers)
			vf_network = FlattenMlp(input_size=self.args.augmented_obs_dim,
									output_size=1,
									hidden_sizes=self.args.dqn_layers)
			policy = TanhGaussianPolicy(obs_dim=self.args.augmented_obs_dim,
										action_dim=self.args.action_dim,
										hidden_sizes=self.args.policy_layers)

			c = FlattenMlp(hidden_sizes=self.args.policy_layers,
							input_size=self.args.augmented_obs_dim + self.args.action_dim,
							output_size=1
						).to(ptu.device)
			self.c_optim = Adam(c.parameters(), lr=self.args.c_lr)
			self._divergence_name = 'kl'
			self._divergence = divergences.get_divergence(name=self._divergence_name, c=c, device=ptu.device)
			if self.args.agent_sac:
				self.agent = SAC(
					policy,
					q1_network,
					q2_network,
					actor_lr=self.args.actor_lr,
					critic_lr=self.args.critic_lr,
					gamma=self.args.gamma,
					tau=self.args.soft_target_tau,
					use_cql=self.args.use_cql if 'use_cql' in self.args else False,
					alpha_cql=self.args.alpha_cql if 'alpha_cql' in self.args else None,
					entropy_alpha=self.args.entropy_alpha,
					automatic_entropy_tuning=self.args.automatic_entropy_tuning,
					alpha_lr=self.args.alpha_lr,
					clip_grad_value=self.args.clip_grad_value,
				).to(ptu.device)
			else:
				self.agent = FOCALSAC(
					policy,
					q1_network,
					q2_network,
					vf_network,
					c,

					actor_lr=self.args.actor_lr,
					critic_lr=self.args.critic_lr,
					vf_lr=self.args.vf_lr,
					c_lr=self.args.c_lr,
					gamma=self.args.gamma,
					tau=self.args.soft_target_tau,
					c_iter=self.args.c_iter,

					rl_batch_size=self.args.rl_batch_size,
					use_cql=self.args.use_cql if 'use_cql' in self.args else False,
					alpha_cql=self.args.alpha_cql if 'alpha_cql' in self.args else None,
					entropy_alpha=self.args.entropy_alpha,
					automatic_entropy_tuning=self.args.automatic_entropy_tuning,
					alpha_lr=self.args.alpha_lr,
					clip_grad_value=self.args.clip_grad_value,
					
					use_iql = self.args.use_iql if 'use_iql' in self.args else False,
					quantile = self.args.quantile,
					beta = self.args.beta,
					clip_score = self.args.clip_score,
				).to(ptu.device)
	# convert the training set to the multitask replay buffer
	def load_buffer(self, train_dataset, train_goals):
		# process obs, actions, ... into shape (num_trajs*num_timesteps, dim) for each task
		dataset = []
		# total_transition_per_task = len(train_dataset[0][0]) * len(train_dataset[0][0][0])
		# visual = np.zeros((len(train_goals), total_transition_per_task, self.args.obs_dim * 2 + self.args.action_dim + 1 + 1))
		for i, set in enumerate(train_dataset):
			obs, actions, rewards, next_obs, terminals = set
			
			device=ptu.device
			obs = ptu.FloatTensor(obs).to(device)
			actions = ptu.FloatTensor(actions).to(device)
			rewards = ptu.FloatTensor(rewards).to(device)
			next_obs = ptu.FloatTensor(next_obs).to(device)
			terminals = ptu.FloatTensor(terminals).to(device)

			obs = obs.transpose(0, 1).reshape(-1, obs.shape[-1])
			actions = actions.transpose(0, 1).reshape(-1, actions.shape[-1])
			rewards = rewards.transpose(0, 1).reshape(-1, rewards.shape[-1])
			next_obs = next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
			terminals = terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])

			obs = ptu.get_numpy(obs)
			actions = ptu.get_numpy(actions)
			rewards = ptu.get_numpy(rewards)
			next_obs = ptu.get_numpy(next_obs)
			terminals = ptu.get_numpy(terminals)

			dataset.append([obs, actions, rewards, next_obs, terminals])
			
			# visual[i] = np.concatenate((obs, actions, rewards, next_obs, terminals),axis=-1)

		# vis_dataset = visual.reshape(len(train_goals) * total_transition_per_task, -1)
		# vis_labels = np.repeat(np.arange(len(train_goals)), total_transition_per_task)
		# visual_by_tsne(vis_dataset, vis_labels, self.args.env_name)
		# exit(0)

		#augmented_obs_dim = dataset[0][0].shape[1]

		self.storage = MultiTaskPolicyStorage(max_replay_buffer_size=self.args.max_replay_size,    #dataset[0][0].shape[0]
											  obs_dim=dataset[0][0].shape[1],
											  action_space=self.args.act_space,
											  tasks=range(len(train_goals)),
											  trajectory_len=self.args.trajectory_len)

		for task, set in enumerate(dataset):  
			self.storage.add_samples(task,
									 observations=set[0],
									 actions=set[1],
									 rewards=set[2],
									 next_observations=set[3],
									 terminals=set[4])  
		return #train_goals, augmented_obs_dim
	

	# training offline RL, with evaluation on fixed eval tasks
	def train(self):
		self._start_training()
		#print('start training')
		for iter_ in range(self.args.num_iters):   
			self.training_mode(True)
			indices = np.random.choice(len(self.goals), self.args.meta_batch) # sample with replacement! it is important for FOCAL
		
			#print('training')
			train_stats = self.update(indices)

			self.training_mode(False)
			#print('logging')
			self.log(iter_ + 1, train_stats)

	# metric loss in FOCAL
	def metric_loss(self, z, tasks, epsilon=1e-3):
		# z shape is (task, corresponding dim)
		pos_z_loss = 0.
		neg_z_loss = 0.
		pos_cnt = 0
		neg_cnt = 0
		for i in range(len(tasks)):
			for j in range(i+1, len(tasks)):
				# positive pair
				if tasks[i] == tasks[j]:
					pos_z_loss += torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
					pos_cnt += 1
				else:
					neg_z_loss += 1/(torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)  
					neg_cnt += 1
		#print(pos_z_loss, pos_cnt, neg_z_loss, neg_cnt)
		return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

	def supervised_loss(self, task_embedding, input_obs, input_action, supervise_next_obs, supervise_reward):

		task_embedding = task_embedding.repeat(len(input_obs), 1, 1).reshape(-1, self.args.task_embedding_size)
		input_obs = input_obs.reshape(-1, self.args.obs_dim)
		input_action = input_action.reshape(-1, self.args.action_dim)
		supervise_next_obs = supervise_next_obs.reshape(-1, self.args.obs_dim)
		supervise_reward = supervise_reward.reshape(-1, 1)
		
		return self.decoder.loss(task_embedding, input_obs, input_action, supervise_next_obs, supervise_reward)

	def update(self, tasks):
		rl_losses_agg = {}
		time_cost = {'data_sampling':0, 'update_encoder':0, 'update_rl':0}

		for update in range(self.args.rl_updates_per_iter): 
			if self.args.log_train_time:
				_t_cost = time.time()
			#print('data sampling')

			obs, actions, rewards, next_obs, terms = self.sample_rl_batch(tasks, self.args.rl_batch_size) # [task, batch, dim]
			
			for _ in range (self.args.decoder_iter):
				if self.args.use_decoder:
					task_encoding, metric_loss = self.finetune_update_encoder(tasks)
				else:
					task_encoding, metric_loss = self.offline_update_encoder(time_cost, tasks)
			
			task_encoding = task_encoding.detach().unsqueeze(1)
			t, _, d = task_encoding.size()
			task_encoding = task_encoding.expand(t, self.args.rl_batch_size, d) # [task, batch(repeat), dim]

			obs = torch.cat((obs, task_encoding), dim=-1)
			next_obs = torch.cat((next_obs, task_encoding), dim=-1) # [task, batch, obs_dim+z_dim]

			# flatten out task dimension
			t, b, _ = obs.size()
			obs = obs.view(t * b, -1)
			actions = actions.view(t * b, -1)
			rewards = rewards.view(t * b, -1)
			next_obs = next_obs.view(t * b, -1)
			terms = terms.view(t * b, -1)

			new_action, _, _, next_log_prob = self.agent.act(obs, return_log_prob=True)
			div_estimate = self._divergence.dual_estimate(
				obs, new_action, actions, False)       
			c_loss = self._divergence.dual_critic_loss(obs, new_action, actions)  
			self.c_optim.zero_grad()
			c_loss.backward(retain_graph=True)
			self.c_optim.step()
			for _ in range(self.args.c_iter - 1):  
				self._optimize_c(indices=tasks, context=task_encoding)

			#print('forward: q learning')
			# RL update (Q learning)   
			rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms, div_estimate=div_estimate, action_space=self.env.action_space)

			rl_losses['metric_loss'] =  metric_loss.item()

			if self.args.log_train_time:
				_t_now = time.time()
				time_cost['update_rl'] += (_t_now-_t_cost)
				_t_cost = _t_now


			for k, v in rl_losses.items():
				if update == 0:  # first iterate - create list
					rl_losses_agg[k] = [v]
				else:  # append values
					rl_losses_agg[k].append(v)
		# take mean
		for k in rl_losses_agg:
			rl_losses_agg[k] = np.mean(rl_losses_agg[k])
		self._n_rl_update_steps_total += self.args.rl_updates_per_iter

		if self.args.log_train_time:
			print(time_cost)

		return rl_losses_agg

	def offline_update_encoder(self, time_cost, tasks):
		# sample corresponding context batch  
		obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch(tasks) # [ts'=ts*num_context_traj, task, dim]

		#self.agent.optimizer.zero_grad()
		self.encoder_optimizer.zero_grad()

		if self.args.log_train_time:
			_t_now = time.time()
			time_cost['data_sampling'] += (_t_now-_t_cost)
			_t_cost = _t_now

		#update context encoder with contrastive loss 
		task_encoding, _ = self.encoder.context_encoding(obs=obs_context, actions=actions_context, 
			rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)
		metric_loss = self.metric_loss(task_encoding, tasks)
		metric_loss.backward()
		self.encoder_optimizer.step()

		if self.args.log_train_time:
			_t_now = time.time()
			time_cost['update_encoder'] += (_t_now-_t_cost)
			_t_cost = _t_now
			
		return task_encoding, metric_loss
	
	def finetune_update_encoder(self, tasks):
     
		# sample corresponding context batch. Here assume to use self.storage to provide context
		obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch_finetune(tasks, self.storage) 

		#update context encoder with contrastive loss   
		task_encoding, encoder_loss = self.encoder.context_encoding(obs=obs_context, actions=actions_context, 
			rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)
		total_loss = self.metric_loss(task_encoding, tasks) * self.args.beta_encoder
		total_loss += self.supervised_loss(task_embedding=task_encoding,
										   input_obs=obs_context,
										   input_action=actions_context,
										   supervise_next_obs=next_obs_context,
										   supervise_reward=rewards_context)
		# total_loss += self.args.beta_encoder * encoder_loss
		self.finetune_encoder_decoder_optimizer.zero_grad()
		total_loss.backward()
		self.finetune_encoder_decoder_optimizer.step()
  
			
		return task_encoding, total_loss
			

	# do policy evaluation on eval tasks
	def evaluate(self, trainset=False, ood=False):
		num_episodes = self.args.max_rollouts_per_task         
		num_steps_per_episode = self.env.unwrapped._max_episode_steps
		num_tasks = self.args.num_train_tasks if trainset else self.args.num_eval_tasks 
		obs_size = self.env.unwrapped.observation_space.shape[0]

		returns_per_episode = np.zeros((num_tasks, num_episodes))
		success_rate = np.zeros(num_tasks)

		rewards = np.zeros((num_tasks, self.args.trajectory_len))
		reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
		observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
		if self.args.policy == 'sac':
			log_probs = np.zeros((num_tasks, self.args.trajectory_len))

		eval_env = self.env_train if trainset else self.env
		
		for task in eval_env.unwrapped.get_all_task_idx():
			obs = ptu.from_numpy(eval_env.reset(task))
			obs = obs.reshape(-1, obs.shape[-1])
			step = 0       
			
			if ood:
				obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_ood_batch([task], trainset=trainset)
			else:
				obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch([task], trainset=trainset)
			#print(obs_context.size())
			# extract task encodings
			'''
			_, mean, logvar, hidden_state = self.encoder.prior(batch_size=obs_context.shape[1])
			for s_ in range(self.args.trajectory_len * self.args.num_context_trajs):
				# update encoding
				_, mean, logvar, hidden_state = self.encoder.forward(
					states=obs_context[s_].unsqueeze(0),
					actions=actions_context[s_].unsqueeze(0),
					rewards=rewards_context[s_].unsqueeze(0),
					hidden_state=hidden_state,
					return_prior=False
				)                
			task_desc = mean # [1, dim]
			'''
			task_desc, _ = self.encoder.context_encoding(obs=obs_context, actions=actions_context, 
				rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)    

			observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

			for episode_idx in range(num_episodes):
				running_reward = 0.
				for step_idx in range(num_steps_per_episode):
					# add distribution parameters to observation - policy is conditioned on posterior
					augmented_obs = torch.cat((obs, task_desc), dim=-1)
					if self.args.policy == 'dqn':
						action, value = self.agent.act(obs=augmented_obs, deterministic=True)
					else:
						action, _, _, log_prob = self.agent.act(obs=augmented_obs,
																deterministic=self.args.eval_deterministic,
																return_log_prob=True)

					# observe reward and next obs  
					next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))
					running_reward += reward.item()
					# done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
					# update encoding
					#task_sample, task_mean, task_logvar, hidden_state = self.update_encoding(obs=next_obs,
					#                                                                         action=action,
					#                                                                         reward=reward,
					#                                                                         done=done,
					#                                                                         hidden_state=hidden_state)
					rewards[task, step] = reward.item()
					#reward_preds[task, step] = ptu.get_numpy(
					#    self.vae.reward_decoder(task_sample, next_obs, obs, action)[0, 0])

					observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])
					if self.args.policy != 'dqn':
						log_probs[task, step] = ptu.get_numpy(log_prob[0])

					if "is_goal_state" in dir(eval_env.unwrapped) and eval_env.unwrapped.is_goal_state():
						success_rate[task] = 1.
					# set: obs <- next_obs
					obs = next_obs.clone()
					step += 1

				returns_per_episode[task, episode_idx] = running_reward

		# reward_preds is 0 here
		if self.args.policy == 'dqn':
			return returns_per_episode, success_rate, observations, rewards, reward_preds
		else:
			return returns_per_episode, success_rate, log_probs, observations, rewards, reward_preds

	def load_parameter(self, agent_addr=None, encoder_addr=None, decoder_addr=None):
		if agent_addr:
			self.agent.load_state_dict(torch.load(agent_addr, map_location="cuda:0"))
		if encoder_addr:
			self.encoder.load_state_dict(torch.load(encoder_addr, map_location="cuda:0"))
		if decoder_addr:
			self.decoder.load_state_dict(torch.load(decoder_addr, map_location="cuda:0"))

	def log(self, iteration, train_stats):
		# --- save model ---
		if iteration % self.args.save_interval == 0:
			save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))
			torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))
			torch.save(self.decoder.state_dict(), os.path.join(save_path, "decoder{0}.pt".format(iteration)))

		if iteration % self.args.log_interval == 0 or iteration == 1:
			if self.args.policy == 'dqn':
				returns, success_rate, observations, rewards, reward_preds = self.evaluate()
				returns_train, success_rate_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset=True)
			# This part is super specific for the Semi-Circle env
			# elif self.args.env_name == 'PointRobotSparse-v0':
			#     returns, success_rate, log_probs, observations, \
			#     rewards, reward_preds, reward_belief, reward_belief_discretized, points = self.evaluate()
			else:   
				returns, success_rate, log_probs, observations, rewards, reward_preds = self.evaluate()
				returns_train, success_rate_train, log_probs_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset=True)

			if self.args.log_tensorboard:
				if self.args.env_name == 'GridBlock-v2':
					tasks_to_vis = np.random.choice(self.args.num_eval_tasks, 5)
					for i, task in enumerate(tasks_to_vis):
						self.env.reset(task)
						self.tb_logger.writer.add_figure('policy_vis/task_{}'.format(i),
													 utl_eval.plot_rollouts(observations[task, :], self.env),
													 self._n_rl_update_steps_total)
						self.tb_logger.writer.add_figure('reward_prediction_train/task_{}'.format(i),
													 utl_eval.plot_rew_pred_vs_rew(rewards[task, :],
																				   reward_preds[task, :]),
													 self._n_rl_update_steps_total)

				if self.args.max_rollouts_per_task > 1:
					raise NotImplementedError
				else:   
					self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/returns_mean', np.mean(returns_train),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/returns_std', np.std(returns_train),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('returns_train/success_rate', np.mean(success_rate_train),
													 self._n_rl_update_steps_total)
				if self.args.policy == 'dqn':
					self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_updates', train_stats['qf_loss'],
													 self._n_rl_update_steps_total)
					# other loss terms
					for k in train_stats.keys():
						if k != 'qf_loss':
							self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
								self._n_rl_update_steps_total)

					self.tb_logger.writer.add_scalar('weights/q_network',
													 list(self.agent.qf.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf.parameters())[0].grad is not None:
						param_list = list(self.agent.qf.parameters())
						self.tb_logger.writer.add_scalar('gradients/q_network',
														 sum([param_list[i].grad.mean() for i in
															  range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q_target',
													 list(self.agent.target_qf.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.target_qf.parameters())[0].grad is not None:
						param_list = list(self.agent.target_qf.parameters())
						self.tb_logger.writer.add_scalar('gradients/q_target',
														 sum([param_list[i].grad.mean() for i in
															  range(len(param_list))]),
														 self._n_rl_update_steps_total)
				else:
					self.tb_logger.writer.add_scalar('policy/log_prob', np.mean(log_probs),
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('rl_losses/qf1_loss', train_stats['qf1_loss'],
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('rl_losses/qf2_loss', train_stats['qf2_loss'],
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('rl_losses/policy_loss', train_stats['policy_loss'],
													 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('rl_losses/alpha_entropy_loss', train_stats['alpha_entropy_loss'],
													 self._n_rl_update_steps_total)

					# other loss terms
					for k in train_stats.keys():
						if k not in ['qf1_loss', 'qf2_loss', 'policy_loss', 'alpha_entropy_loss']:
							self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
								self._n_rl_update_steps_total)

					# weights and gradients
					self.tb_logger.writer.add_scalar('weights/q1_network',
													 list(self.agent.qf1.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf1.parameters())[0].grad is not None:
						param_list = list(self.agent.qf1.parameters())
						self.tb_logger.writer.add_scalar('gradients/q1_network',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q1_target',
													 list(self.agent.qf1_target.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf1_target.parameters())[0].grad is not None:
						param_list = list(self.agent.qf1_target.parameters())
						self.tb_logger.writer.add_scalar('gradients/q1_target',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q2_network',
													 list(self.agent.qf2.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf2.parameters())[0].grad is not None:
						param_list = list(self.agent.qf2.parameters())
						self.tb_logger.writer.add_scalar('gradients/q2_network',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/q2_target',
													 list(self.agent.qf2_target.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.qf2_target.parameters())[0].grad is not None:
						param_list = list(self.agent.qf2_target.parameters())
						self.tb_logger.writer.add_scalar('gradients/q2_target',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)
					self.tb_logger.writer.add_scalar('weights/policy',
													 list(self.agent.policy.parameters())[0].mean(),
													 self._n_rl_update_steps_total)
					if list(self.agent.policy.parameters())[0].grad is not None:
						param_list = list(self.agent.policy.parameters())
						self.tb_logger.writer.add_scalar('gradients/policy',
														 sum([param_list[i].grad.mean() for i in range(len(param_list))]),
														 self._n_rl_update_steps_total)

			print("Iteration -- {}, Success rate -- {:.3f}, Avg. return -- {:.3f}, \
				Success rate train -- {:.3f}, Avg. return train -- {:.3f}, Elapsed time {:5d}[s]"
				  .format(iteration, np.mean(success_rate), np.mean(np.sum(returns, axis=-1)),
					np.mean(success_rate_train), np.mean(np.sum(returns_train, axis=-1)), 
						  int(time.time() - self._start_time)), train_stats)

	def sample_rl_batch(self, tasks, batch_size):
		''' sample batch of unordered rl training data from a list/array of tasks '''
		# this batch consists of transitions sampled randomly from replay buffer
		batches = [ptu.np_to_pytorch_batch(
			self.storage.random_batch(task, batch_size)) for task in tasks]
		unpacked = [utl.unpack_batch(batch) for batch in batches]
		# group elements together
		unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
		unpacked = [torch.cat(x, dim=0) for x in unpacked]
		return unpacked
	
	def sample_context_batch_finetune(self, tasks, context_buffer: MultiTaskPolicyStorage):
		# context buffer used to generate the context !
		# immitate the function of sample_context_batch
		context = []
		for i in tasks:
			context_dict_i = context_buffer.random_trajectories(i, self.args.num_context_trajs)
			context_tensor_i = [torch.FloatTensor(context_dict_i['observations']).to(ptu.device), 
								torch.FloatTensor(context_dict_i['actions']).to(ptu.device), 
								torch.FloatTensor(context_dict_i['rewards']).to(ptu.device),
								torch.FloatTensor(context_dict_i['next_observations']).to(ptu.device),
								torch.FloatTensor(context_dict_i['terminals']).to(ptu.device)]
			context.append(context_tensor_i)
		ret = [torch.stack([context[i][j] for i in range(len(tasks))], dim=0).transpose(0,1) for j in range(5)]
		return ret 


	# sample num_context_trajs trajectories in buffer for each task, as task context
	# trainset: if true, tasks are in context_dataset, else, tasks are in eval_context_dataset
	def sample_context_batch(self, tasks, trainset=True):
		if trainset:
			contextset = self.context_dataset
		else:
			contextset = self.eval_context_dataset


		context = []
		for i in tasks:
			i_episodes = np.random.choice(contextset[i][0].shape[1], self.args.num_context_trajs) # should be randomized at every task
			context_i = [ptu.FloatTensor(contextset[i][j][:, i_episodes, :]).transpose(0,1).reshape(
				-1, contextset[i][j].shape[-1]) for j in range(len(contextset[i]))] # obs, act, reward, next_obs, term 
			context.append(context_i)

		ret = [torch.stack([context[i][j] for i in range(len(tasks))], dim=0).transpose(0,1) for j in range(len(contextset[i]))]
		return ret 
	
	def sample_ood_batch(self, tasks, trainset=True):
		test_env = make_env(self.args.env_name,
			self.args.max_rollouts_per_task,
			seed=self.args.seed,
			n_tasks=1)

		if trainset:
			goals = self.goals[tasks]
		else:
			goals = self.eval_goals[tasks]
		#print(goals)

		context = []
		for i, g in enumerate(goals):
			obs_c, act_c, rew_c, next_obs_c, term_c = [],[],[],[],[]

			for rollout in range(self.args.num_context_trajs):
				test_env.set_goal(g)
				obs = ptu.from_numpy(test_env.reset())
				obs = obs.reshape(-1, obs.shape[-1])
				done_rollout = False

				while not done_rollout:
					if self.args.policy == 'dqn':
						action, _ = self.context_agent.act(obs=obs)   # DQN
					else:
						action, _, _, _ = self.context_agent.act(obs=obs)   # SAC
					# observe reward and next obs
					next_obs, reward, done, info = utl.env_step(test_env, action.squeeze(dim=0))
					done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True

					# add data to policy buffer - (s+, a, r, s'+, term)
					term = test_env.unwrapped.is_goal_state() if "is_goal_state" in dir(test_env.unwrapped) else False
					rew_to_buffer = ptu.get_numpy(reward.squeeze(dim=0))
					
					obs_c.append(ptu.get_numpy(obs.squeeze(dim=0)))
					act_c.append(ptu.get_numpy(action.squeeze(dim=0)))
					next_obs_c.append(ptu.get_numpy(next_obs.squeeze(dim=0)))
					rew_c.append(rew_to_buffer)
					term_c.append(np.array([term], dtype=float))

					# set: obs <- next_obs
					obs = next_obs.clone()

			obs_c = ptu.FloatTensor(np.stack(obs_c))
			act_c = ptu.FloatTensor(np.stack(act_c))
			rew_c = ptu.FloatTensor(np.stack(rew_c))
			next_obs_c = ptu.FloatTensor(np.stack(next_obs_c))
			term_c = ptu.FloatTensor(np.stack(term_c))
			#print(obs_c.shape, act_c.shape, rew_c.shape, next_obs_c.shape, term_c.shape)

			context_i = [obs_c, act_c, rew_c, next_obs_c, term_c]
			context.append(context_i)

		ret = [torch.stack([context[i][j] for i in range(len(tasks))], dim=0).transpose(0,1) for j in range(5)]
		#print(ret[0].shape)
		#sys.exit(0)
		return ret

	def _optimize_c(self, indices, context):
		# data is (task, batch, feat)
		obs, actions, rewards, next_obs, terms = self.sample_rl_batch(indices, self.args.rl_batch_size) # [task, batch, dim]
		obs = torch.cat((obs, context), dim=-1)

		# flattens out the task dimension
		t, b, _ = obs.size()
		obs = obs.view(t * b, -1)
		actions = actions.view(t * b, -1)
		next_obs = next_obs.view(t * b, -1)

		# run inference in networks
		new_actions, _, _, next_log_prob = self.agent.act(obs, return_log_prob=True)

		# optimize for c network (which computes dual-form divergences)
		c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions)
		self.c_optim.zero_grad()
		c_loss.backward(retain_graph=True)
		self.c_optim.step()

	def _start_training(self):
		self._n_rl_update_steps_total = 0
		self._start_time = time.time()

	def training_mode(self, mode):
		self.agent.train(mode)
		self.encoder.train(mode)

	def load_behavior_policy(self, path):
		q1_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
								output_size=1,
								hidden_sizes=[128,128])
		q2_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
								output_size=1,
								hidden_sizes=[128,128])
		policy = TanhGaussianPolicy(obs_dim=self.args.obs_dim,
									action_dim=self.args.action_dim,
									hidden_sizes=[128,128])
		self.context_agent = SAC(
			policy,
			q1_network,
			q2_network,

			actor_lr=self.args.actor_lr,
			critic_lr=self.args.critic_lr,
			gamma=self.args.gamma,
			tau=self.args.soft_target_tau,

			entropy_alpha=self.args.entropy_alpha,
			automatic_entropy_tuning=self.args.automatic_entropy_tuning,
			alpha_lr=self.args.alpha_lr
		).to(ptu.device)
		
		self.context_agent.load_state_dict(torch.load(path, map_location="cuda:0"))



def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--env-type', default='gridworld')
	# parser.add_argument('--env-type', default='point_robot_v1')
	parser.add_argument('--env-type', default='walker_param')
	# parser.add_argument('--env-type', default='cheetah_vel')
	# parser.add_argument('--env-type', default='ant_dir')
	# parser.add_argument('--env-type', default='grid_block')
	args, rest_args = parser.parse_known_args()
	env = args.env_type

	if env == 'cheetah_vel':
		args = args_cheetah_vel.get_args(rest_args)
	elif env == 'cheetah_dir':
		args = args_cheetah_dir.get_args(rest_args)
	elif env == 'ant_dir':
		args = args_ant_dir.get_args(rest_args)
	elif env == 'hopper_param':
		args = args_hopper_param.get_args(rest_args)
	elif env == 'walker_param':
		args = args_walker_param.get_args(rest_args)
	else:
		raise NotImplementedError

	print(args.use_gpu)
	set_gpu_mode(torch.cuda.is_available())
	print(ptu.device)

	#vae_args = config_utl.load_config_file(os.path.join(args.vae_dir, args.env_name,
	#                                                    args.vae_model_name, 'online_config.json'))
	#args = config_utl.merge_configs(vae_args, args)     # order of input to this function is important
	#print(args)
	args, _ = off_utl.expand_args(args) # add env information to args
	#print(args)


	dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
	assert args.num_train_tasks + args.num_eval_tasks == len(goals)
	train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
	eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]
 

	learner = FOCAL(args, train_dataset, train_goals, eval_dataset, eval_goals)
 
	learner.train()
	
	OOD_ret = np.array([])

	with open('ood_test_config/{}.txt'.format(args.env_name), 'r') as f:
		behavior_policy_list = f.read().splitlines()
	
	with open('ood_test_result/{}.txt'.format(args.env_name + "_" + str(args.seed) + "_" + str(args.beta_encoder)), "w") as file:
		file.close()
  
	ood_ret = np.array([])
	for l in behavior_policy_list:
		learner.load_behavior_policy(path=l) # a random agent trained on another task to collect context
		#print('context collection policy loaded')

		returns, success_rate, log_probs, observations, rewards, reward_preds = learner.evaluate(ood=True)
		#print("ood returns", returns)
		with open('ood_test_result/{}.txt'.format(args.env_name + "_" + str(args.seed) + "_" + str(args.beta_encoder)), "a") as file:
			print('behavior policy: {}, return: {} std:{}'.format(l, np.mean(returns), np.std(returns)), file=file)
			file.write("\n")
		ood_ret = np.append(ood_ret, np.mean(returns))
	
	with open('ood_test_result/{}.txt'.format(args.env_name + "_" + str(args.seed) + "_" + str(args.beta_encoder)), "a") as file:
		print('mean ood return:', np.mean(ood_ret), file=file)
		file.write("\n")
	OOD_ret = np.append(OOD_ret, np.mean(ood_ret))
	print()
	print('OOD: mean {}, std {}'.format(np.mean(OOD_ret), np.std(OOD_ret)))


if __name__ == '__main__':
	main()
