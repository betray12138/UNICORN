
import os
from collections import OrderedDict
from torchkit.eval_util import create_stats_ordered_dict
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchkit.pytorch_utils as ptu
from torchkit.brac import divergences
from utils import helpers as utl

class FOCALSAC(nn.Module):
    def __init__(self,
                 policy,
                 q1_network,
                 q2_network,
                 vf,
                 c,

                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 vf_lr=3e-4,
                 c_lr=1e-4,
                 gamma=0.99,
                 tau=5e-3,
                 c_iter=3,

                 rl_batch_size=256,
                 use_cql=False,
                 use_iql=False,
                 quantile = 0.5,
                 beta=0.1,
                 clip_score=100,
                 alpha_cql=2.,
                 entropy_alpha=0.2,
                 automatic_entropy_tuning=True,
                 alpha_lr=3e-4,
                 clip_grad_value=None
                 ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.use_cql = use_cql    # Conservative Q-Learning loss
        self.use_iql = use_iql    # Implicit Q-Learning loss
        self.quantile = quantile
        self.beta = beta
        self.cilp_score = clip_score
        self.alpha_cql = alpha_cql    # Conservative Q-Learning weight parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning    # Wasn't tested
        self.clip_grad_value = clip_grad_value

        # q networks - use two network to mitigate positive bias
        self.qf1 = q1_network
        self.qf1_optim = Adam(self.qf1.parameters(), lr=critic_lr)

        self.qf2 = q2_network
        self.qf2_optim = Adam(self.qf2.parameters(), lr=critic_lr)

        # target networks
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.vf = vf
        self.vf_optim = Adam(self.vf.parameters(), lr=vf_lr)

        self.policy = policy
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

        self.c = c
        self.c_optim = Adam(self.c.parameters(), lr=c_lr)

        self._divergence_name = 'kl'
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

        self.rl_batch_size = rl_batch_size
        
        self.eval_statistics = None
        self.z_means = None
        self.z_vars = None
        self.z_loss = None
        self.task_id = None

        self._c_iter = c_iter
        self.storage = None

        # automatic entropy coefficient tuning
        if self.automatic_entropy_tuning:
            # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(ptu.device)).item()
            self.target_entropy = -self.policy.action_dim
            self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=ptu.device)
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp()
        else:
            self.alpha_entropy = entropy_alpha

    def forward(self, obs):
        action, _, _, _ = self.policy(obs)
        q1, q2 = self.qf1(obs, action), self.qf2(obs, action)
        return action, q1, q2

    def act(self, obs, deterministic=False, return_log_prob=False):
        action, mean, log_std, log_prob = self.policy(obs,
                                                      deterministic=deterministic,
                                                      return_log_prob=return_log_prob)
        return action, mean, log_std, log_prob

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(ptu.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, obs, action, reward, next_obs, done, div_estimate, **kwargs):
        
        # computation of critic loss
        #torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            if self.use_iql:
                q_target_pred_iql = torch.min(
                        self.qf1_target(obs, action),
                        self.qf2_target(obs, action),
                    )
                vf_pred_next_state = self.vf(next_obs)
                q_target = reward + (1. - done) * self.gamma * vf_pred_next_state
            else:
                next_action, _, _, next_log_prob = self.act(next_obs, return_log_prob=True)
                next_q1 = self.qf1_target(next_obs, next_action)
                next_q2 = self.qf2_target(next_obs, next_action)
                min_next_q_target = torch.min(next_q1, next_q2) - self.alpha_entropy * div_estimate
                q_target = reward + (1. - done) * self.gamma *  min_next_q_target

        q1_pred = self.qf1(obs, action)
        q2_pred = self.qf2(obs, action)

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # use CQL loss for offline RL (Kumar et al, 2020)
        if self.use_cql and ~self.use_iql:
            qf1_loss += torch.mean(self.alpha_cql *
                                   self.estimate_log_sum_exp_q(self.qf1, obs, N=10, action_space=kwargs['action_space'])
                                   - q1_pred)
            qf2_loss += torch.mean(self.alpha_cql *
                                   self.estimate_log_sum_exp_q(self.qf2, obs, N=10, action_space=kwargs['action_space'])
                                   - q2_pred)
            
        # update vf network
        if self.use_iql: 
            vf_pred = self.vf(obs)
            vf_err = vf_pred - q_target_pred_iql
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (vf_err ** 2)).mean()
            self.vf_optim.zero_grad()
            vf_loss.backward()
            if self.clip_grad_value is not None:
                self._clip_grads(self.vf)
            self.vf_optim.step()
            
        
        # update q networks
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf1)
            self._clip_grads(self.qf2)
        self.qf1_optim.step()
        self.qf2_optim.step()
        # soft update
        self.soft_target_update()


        # computation of actor loss
        # bug fixed: this should be after the q function update
        new_action, policy_mean, policy_log_std, log_prob = self.act(obs, return_log_prob=True)

        if self.use_iql:
            with torch.no_grad():
                adv = q_target_pred_iql - vf_pred
                exp_adv = torch.exp(adv / self.beta)
                if self.cilp_score is not None:
                    exp_adv = torch.clamp(exp_adv, max=self.cilp_score)
                policy_weight = exp_adv[:, 0]
            policy_loss = (-log_prob * policy_weight).mean()
        else:
            min_q_new_actions = self._min_q(obs, new_action)
            # policy_loss = ((self.alpha_entropy * log_prob) - min_q_new_actions).mean()     
            policy_loss = (log_prob - min_q_new_actions + self.alpha_entropy * div_estimate).mean() 
            

        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.policy)
        self.policy_optim.step()

        if ~self.use_iql: 
            if self.automatic_entropy_tuning:
                alpha_entropy_loss = -(self.log_alpha_entropy * (log_prob + self.target_entropy).detach()).mean()

                self.alpha_entropy_optim.zero_grad()
                alpha_entropy_loss.backward()
                self.alpha_optim.step()

                self.alpha_entropy = self.log_alpha_entropy.exp()
                # alpha_entropy_tlogs = self.alpha_entropy.clone()    # For TensorboardX logs
            else:
                alpha_entropy_loss = torch.tensor(0.).to(ptu.device)
                # alpha_entropy_tlogs = torch.tensor(self.alpha_entropy)  # For TensorboardX logs

        if self.eval_statistics is None and self.z_means is not None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            for i in range(len(self.z_means[0])):
                z_mean = ptu.get_numpy(self.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean
            z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
            
            if self.use_iql:
                self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            
            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['Z Loss'] = ptu.get_numpy(self.z_loss)
            self.eval_statistics['task idx'] = self.task_id
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['div_estimate'] = np.mean(ptu.get_numpy(div_estimate))
            # self.eval_statistics['C Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_prob)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha',          self.alpha_entropy))

        return {'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item(),
                'policy_loss': policy_loss.item(), 'alpha_entropy_loss': alpha_entropy_loss.item(), 'q_target': torch.mean(q_target).item(),
                'qf1_pred': torch.mean(q1_pred).item(), 'qf2_pred': torch.mean(q2_pred).item(), 'log_prob': torch.mean(log_prob).item()}

    def _min_q(self, obs, action):
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        min_q = torch.min(q1, q2)
        return min_q

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)

    def _clip_grads(self, net):
        for p in net.parameters():
            p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)

    def estimate_log_sum_exp_q(self, qf, obs, N, action_space):
        '''
            estimate log(sum(exp(Q))) for CQL objective
        :param qf: Q function
        :param obs: state batch from buffer (s~D)
        :param N: number of actions to sample for estimation
        :param action_space: space of actions -- for uniform sampling
        :return:
        '''
        batch_size = obs.shape[0]
        obs_rep = obs.repeat(N, 1)

        # draw actions at uniform
        random_actions = ptu.FloatTensor(np.vstack([action_space.sample() for _ in range(N)]))
        random_actions = torch.repeat_interleave(random_actions, batch_size, dim=0)
        unif_a = 1 / np.prod(action_space.high - action_space.low)  # uniform density over action space

        # draw actions from current policy
        with torch.no_grad():
            policy_actions, _, _, policy_log_probs = self.act(obs_rep, return_log_prob=True)

        exp_q_unif = qf(obs_rep, random_actions) / unif_a
        exp_q_policy = qf(obs_rep, policy_actions) / torch.exp(policy_log_probs)
        log_sum_exp = torch.log(0.5 * torch.mean((exp_q_unif + exp_q_policy).reshape(N, batch_size, -1), dim=0))

        return log_sum_exp

    def _optimize_c(self, indices):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_rl_batch(indices, self.rl_batch_size) # [task, batch, dim]

        # run inference in networks
        
        policy_outputs, task_z, task_z_vars = self.act(obs, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions)
        self.c_optim.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optim.step()

    # # Save model parameters
    # def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
    #     if not os.path.exists('models/'):
    #         os.makedirs('models/')
    #
    #     if actor_path is None:
    #         actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
    #     if critic_path is None:
    #         critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
    #     print('Saving models to {} and {}'.format(actor_path, critic_path))
    #     torch.save(self.policy.state_dict(), actor_path)
    #     torch.save(self.critic.state_dict(), critic_path)
    #
    # # Load model parameters
    # def load_model(self, actor_path, critic_path):
    #     print('Loading models from {} and {}'.format(actor_path, critic_path))
    #     if actor_path is not None:
    #         self.policy.load_state_dict(torch.load(actor_path))
    #     if critic_path is not None:
    #         self.critic.load_state_dict(torch.load(critic_path))

    def update_critic(self, obs, action, reward, next_obs, done, **kwargs):
        # computation of critic loss
        #torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            next_action, _, _, next_log_prob = self.act(next_obs, return_log_prob=True)
            next_q1 = self.qf1_target(next_obs, next_action)
            next_q2 = self.qf2_target(next_obs, next_action)
            min_next_q_target = torch.min(next_q1, next_q2) - self.alpha_entropy * next_log_prob
            q_target = reward + (1. - done) * self.gamma * min_next_q_target


        q1_pred = self.qf1(obs, action)
        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        if self.use_cql:
            qf1_loss += torch.mean(self.alpha_cql *
                                   self.estimate_log_sum_exp_q(self.qf1, obs, N=10, action_space=kwargs['action_space'])
                                   - q1_pred)
        self.qf1_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf1)
        self.qf1_optim.step()


        q2_pred = self.qf2(obs, action)
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error
        # use CQL loss for offline RL (Kumar et al, 2020)
        if self.use_cql:
            qf2_loss += torch.mean(self.alpha_cql *
                                   self.estimate_log_sum_exp_q(self.qf2, obs, N=10, action_space=kwargs['action_space'])
                                   - q2_pred)
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.qf2)
        self.qf2_optim.step()
        
        # soft update
        self.soft_target_update()

        return {'qf1_loss': qf1_loss.item(), 'qf2_loss': qf2_loss.item()}



    def update_actor(self, obs, action, reward, next_obs, done, **kwargs):
        # computation of actor loss
        # bug fixed: this should be after the q function update
        new_action, _, _, log_prob = self.act(obs, return_log_prob=True)
        min_q_new_actions = self._min_q(obs, new_action)
        policy_loss = ((self.alpha_entropy * log_prob) - min_q_new_actions).mean()


        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.clip_grad_value is not None:
            self._clip_grads(self.policy)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -(self.log_alpha_entropy * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_optim.step()

            self.alpha_entropy = self.log_alpha_entropy.exp()
            # alpha_entropy_tlogs = self.alpha_entropy.clone()    # For TensorboardX logs
        else:
            alpha_entropy_loss = torch.tensor(0.).to(ptu.device)
            # alpha_entropy_tlogs = torch.tensor(self.alpha_entropy)  # For TensorboardX logs

        return {'policy_loss': policy_loss.item(), 'alpha_entropy_loss': alpha_entropy_loss.item()}