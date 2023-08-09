
import torch
import numpy as np
from config import global_args as g_args
from algorithms.mappo.actor import R_Actor
from algorithms.mappo.critic import R_Critic
from utils.util import prettyprint, update_linear_schedule


class Policy:
    def __init__(self, obs_space, cent_obs_space, act_space):
        self.device = torch.device(g_args("device")) 
        self.lr = g_args("lr")
        self.critic_lr = g_args("critic_lr")
        self.opti_eps = g_args("opti_eps")
        self.weight_decay = g_args("weight_decay")
        self.thread_num = g_args("thread_num")
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.actor = R_Actor(self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.lr, eps=self.opti_eps,weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr,eps=self.opti_eps,weight_decay=self.weight_decay)                        

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        """
        从输入计算动作
        """
        available_actions = np.concatenate(available_actions, dtype=float)
        actions, action_probs= self.actor(obs, available_actions, deterministic)
        return actions, action_probs

    def get_values(self, cent_obs):
        """
        获得值函数的预估
        """
        _pre_process = lambda o: np.array(np.split(_t2n(o), self.thread_num))
        value = self.critic(cent_obs)
        values = _pre_process(value)
        return values
      
    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None):
        """
        获得动作的概率和熵
        """
        action_probs, dist_entropy = self.actor.evaluate_actions(obs, action, available_actions, active_masks)
        return action_probs, dist_entropy

    def get_actor_policy(self):
        return self.actor

    def get_critic_policy(self):
        return self.critic

    def load_actor_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def load_critic_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict)

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

def _t2n(x):
    return x.detach().cpu().numpy()
