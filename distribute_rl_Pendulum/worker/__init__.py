# -*- coding: utf-8 -*-
import os
from torch import optim, nn 
from time import sleep
import traceback
import torch
import numpy as np
from copy import copy
#from train import env as multi_scenario
import distribute_rl_Pendulum.mempool as MEMPOOL
from log import LogDebug, LogErr, LogExc


import gym
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical
#import ray

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else'cpu') 
from models import Pi_net, V_net, ActorNet
#device = torch.device('cpu')
#@ray.remote




class Worker(object):
    """
    Worker 从redis拉取模型，采样数据，把数据写回到mempool
    """
    def __init__(self, args):
        self.pid = os.getpid()
        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.clip = args.clip
        self.K_epochs = args.epoch
        self.betas = args.betas
        self.ent_coef = args.ent_coef
        self.env = gym.make('Pendulum-v1')
        self.bound = self.env.action_space.high[0]

        self.policy = ActorNet(self.env.observation_space.shape[0], self.bound)
        
        redis_config ={}
        redis_config["host"] = self.args.redis_ip
        redis_config["password"] = self.args.redis_pass_word
        redis_config["port"] = self.args.redis_port
        self.redis = MEMPOOL.CreateMemPool(redis_config)
        print("create worker "+ str(self.pid))
        
    def choose_action(self, s):
        s = torch.FloatTensor(s)
        mu, sigma = self.policy(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return np.clip(action, -self.bound, self.bound)

    def run(self):
        LogDebug("begin to run [%s]", self.pid)
        #num_updates = self.args.max_episode // (self.args.num_workers)
        push_interval = self.args.update_interval // (self.args.num_workers)
        timestep = 0
        episode = {}
        while True:
            obs = self.env.reset()
            for t in range(self.args.nsteps):
                trajectory = {}
                action = self.choose_action(obs)

                trajectory["obs"] = obs
                trajectory["action"] = np.array(action)

                obs_, reward, done, _ = self.env.step([action])

                trajectory["next_obs"] = obs_
                trajectory["reward"] = (reward + 8) / 8
                trajectory["done"] = done

                obs = obs_
                timestep += 1
                episode[t] = trajectory
                if timestep >= push_interval:
                    self.send_sample(episode)
                    episode = {}
                    timestep = 0
                    self.wait_to_update_latest_model()

                if done:
                    break



    def send_sample(self, sample_info):
        ret = self.redis.push_sample(sample_info)
        #LogDebug("pid[%s] send_sample ret[%s]", self.pid, ret)
    
    def wait_to_update_latest_model(self):
        self.model_name = "yh_Pendulum_model"
        while True:
            robj = self.redis.pull_model(0, self.model_name)
            if (not robj): #or self.redis.get_sample_list_len():
                LogDebug("Wait fot new model.")
                sleep(3)
            else:
                break
        model_data = robj.data
        self.policy.load_state_dict(model_data)
        #LogDebug(f"worker[{self.args.env_name}] model[{self.model_name}] update finish")