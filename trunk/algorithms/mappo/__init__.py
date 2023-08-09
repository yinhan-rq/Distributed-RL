import numpy as np
import torch
import torch.nn as nn
from utils.util import huber_loss
from algorithms.utils.util import check
from config import global_args as g_args
from .policy import Policy

class MAPPO:

    def __init__(self, obs_space, cent_obs_space, act_space):
        self.device = g_args("device")
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.policy = Policy(obs_space, cent_obs_space, act_space)
        self.clip_param = g_args("clip_param")
        self.ppo_epoch = g_args("ppo_epoch")
        self.num_mini_batch = g_args("num_mini_batch")
        self.data_chunk_length = g_args("data_chunk_length")
        self.value_loss_coef = g_args("value_loss_coef")
        self.entropy_coef = g_args("entropy_coef")
        self.max_grad_norm = g_args("max_grad_norm")
        self.huber_delta = g_args("huber_delta")     

    def get_values(self, cent_obs):
        return self.policy.get_values(cent_obs)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        计算值函数的损失值
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_original = return_batch - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        更新 actor 和 critic 的网络
        """
        share_obs = sample["share_obs_batch"]
        obs = sample["obs_batch"]
        actions = sample["actions_batch"]
        available_actions = sample["available_actions_batch"]
        active_masks = sample["active_masks_batch"]
        active_masks = check(active_masks).to(**self.tpdv)
        action_probs, dist_entropy = self.policy.evaluate_actions(obs, actions, available_actions, active_masks)
        values = self.policy.critic(share_obs)

        value_preds = sample["value_preds_batch"]
        returns = sample["return_batch"]
        old_action_probs = sample["old_action_log_probs_batch"]
        adv_targ= sample["adv_targ"]

        old_action_probs = check(old_action_probs).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds = check(value_preds).to(**self.tpdv)
        returns = check(returns).to(**self.tpdv)

        # 更新actor
        imp_weights = torch.exp(action_probs - old_action_probs)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks).sum() / active_masks.sum()

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()

        # 更新critic
        value_loss = self.cal_value_loss(values, value_preds, returns, active_masks)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param update_actor: (bool) whether to update actor network.
        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def init_data_generator(self, buffer, advantages):
        data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
        self.data_generator = data_generator

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def lr_decay(self, episode, total_episode):
        self.policy.lr_decay(episode, total_episode)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        return self.policy.get_actions(obs, available_actions, deterministic)
   
    def get_values(self, cent_obs):
        return self.policy.get_values(cent_obs)

    def get_actor_policy(self):
        return self.policy.get_actor_policy()

    def get_critic_policy(self):
        return self.policy.get_critic_policy()

    def load_actor_state_dict(self, state_dict):
        self.policy.load_actor_state_dict(state_dict)

    def load_critic_state_dict(self, state_dict):
        self.policy.load_critic_state_dict(state_dict)