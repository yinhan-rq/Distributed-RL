import os
import torch
import numpy as np
from algorithms.mappo import MAPPO
from config import global_args as g_args
from buffer import ExpertBuffer
from algorithms.algorithm.gail import GAIL
from env.chooseenv import make_parallel_env 
from modules.discrimnator import GAILDiscrim
from utils.util import _t2n, concate_observation_from_raw, prettyprint
from utils.encoder import *


class MAPPOLearner:
    def __init__(self):
        self.use_gail = g_args("use_gail")
        self.device = g_args("device")
        self.init()
        
    def init(self):
        tmp_env = make_parallel_env()
        self.obs_space = tmp_env.observation_space
        self.private_obs_space = tmp_env.private_observation_space
        self.act_space = tmp_env.action_space
        self.left_agent_num = g_args("left_agent_num")
        self.thread_num = g_args("thread_num")
        cent_obs_space = tmp_env.share_observation_space[0] if g_args("use_centralized_V") else tmp_env.observation_space
        tmp_env.close()
        self.algor = MAPPO(self.obs_space, cent_obs_space, self.act_space)
        self.IL_buffer = ""
        if(self.use_gail):
            print(self.use_gail, g_args("use_gail"))
            IL_buffer_path = g_args("IL_buffer_path")
            self.set_imitation_buffer(IL_buffer_path)
            self.D = GAILDiscrim(self.obs_space, self.private_obs_space, device = self.device)
            self.gail_trainer = GAIL(self.D)

        #TODO:拿到外面来
    def set_imitation_buffer(self, path):
        expert_buffer_size = g_args("expert_buffer_size")
        agent_num = self.left_agent_num
        obs_shape = self.obs_space.shape
        pri_obs_space = self.private_obs_space.shape
        self.IL_buffer = ExpertBuffer(expert_buffer_size, agent_num, obs_shape, pri_obs_space, path = path)

    def learn(self, buffer):
        if(self.use_gail):
            gail_info = self.train_gail(buffer, self.IL_buffer)
        self.compute(buffer)
        info = self.train(buffer)
        if(self.use_gail):
            info.update(gail_info)
        return info

    @torch.no_grad()
    def compute(self, buffer):
        cent_obs = buffer.get_share_obs(-1)
        self.algor.prep_rollout()
        next_values = self.algor.get_values(cent_obs)
        buffer.compute_returns(next_values)

    def train(self, buffer):
        self.algor.prep_training()
        train_infos = self.algor.train(buffer)
        buffer.after_update()
        return train_infos

    def train_gail(self, buffer, IL_buffer):
        train_infos = self.gail_trainer.train(buffer, IL_buffer)
        return train_infos

    def lr_decay(self, episode, total_episode):
        self.algor.lr_decay(episode, total_episode)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        _pre_process = lambda o: np.array(np.split(_t2n(o), self.thread_num))
        actions, action_probs = self.algor.get_actions(obs, available_actions, deterministic)
        actions = _pre_process(actions)
        action_probs = _pre_process(action_probs)
        return actions, action_probs

    def get_actions_from_raw_obs(self, obs):
        feature_encoder = FeatureEncoder()
        obs = concate_observation_from_raw(feature_encoder.encode(obs))
        obs = np.array(obs, ndmin=3)
        available_actions = np.array(obs[..., :19], ndmin=3)
        obs = np.concatenate(obs)
        action, _ = self.algor.get_actions(obs, available_actions)
        action_final = [0 for _ in range(19)]
        idx = action[0][0]
        action_final[idx] = 1
        return [action_final]

    def get_values(self, cent_obs):
        return self.algor.get_values(cent_obs)
 
    def save_models(self, path):
        if not os.path.exists(path): 
            os.makedirs(path)
        # 保存actor和critic的网络
        policy_actor = self.algor.get_actor_policy()
        torch.save(policy_actor.state_dict(), path+"/actor.pt")
        policy_critic = self.algor.get_critic_policy()
        torch.save(policy_critic.state_dict(), path+"/critic.pt")

    def load_models(self, path):
        # 加载actor和critic的网络
        actor_state_dict = torch.load(f"{path}/actor.pt")
        self.algor.load_actor_state_dict(actor_state_dict)
        critic_state_dict = torch.load(f"{path}/critic.pt")
        self.algor.load_critic_state_dict(critic_state_dict)

    def load_dict(self, dict):
        if not dict:
            return
        self.algor.load_actor_state_dict(dict["actor"])
        self.algor.load_critic_state_dict(dict["critic"])

    def save_dict(self):
        d = {}
        policy_actor = self.algor.get_actor_policy()
        policy_critic = self.algor.get_critic_policy()
        d["actor"] = policy_actor.state_dict()
        d["critic"] = policy_critic.state_dict()
        return d

    def prep_training(self):
        self.algor.prep_training()

    def prep_rollout(self):
        self.algor.prep_rollout()