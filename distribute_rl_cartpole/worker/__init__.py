# -*- coding: utf-8 -*-
import os
from torch import optim, nn 
from time import sleep
import traceback
import torch
import numpy as np
from copy import copy
#from train import env as multi_scenario
import distribute_rl_cartpole.mempool as MEMPOOL
from log import LogDebug, LogErr, LogExc


import gym
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical
#import ray

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else'cpu') 
from models import Policy
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
        self.env = gym.make('CartPole-v1')

        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optim = optim.Adam(self.policy.parameters(), lr=args.lr, betas=self.betas)

        self.loss_fn = nn.MSELoss()
        redis_config ={}
        redis_config["host"] = self.args.redis_ip
        redis_config["password"] = self.args.redis_pass_word
        redis_config["port"] = self.args.redis_port
        self.redis = MEMPOOL.CreateMemPool(redis_config)
        print("create worker "+ str(self.pid))
        

    def run(self):
        LogDebug("begin to run [%s]", self.pid)
        #num_updates = self.args.max_episode // (self.args.num_workers)
        push_interval = self.args.update_interval // (self.args.num_workers)
        timestep = 0
        episode = {}

        while True:
            obs = self.env.reset()
            for step in range(self.args.nsteps):
                trajectory = {}
                with torch.no_grad():
                    action, log_prob = self.policy.act(torch.tensor(obs))
                
                trajectory["obs"] = torch.tensor(obs)
                trajectory["action"] = torch.tensor(action)
                trajectory["old_probs"] = log_prob

                obs, reward, done, _ = self.env.step(action)

                trajectory["reward"] = reward
                trajectory["done"] = done

                episode[timestep] = trajectory
                timestep += 1

                if timestep >= push_interval:
                    self.send_sample(episode)
                    episode = {}
                    timestep = 0
                    self.wait_to_update_latest_model()

                if done:
                    #obs = self.env.reset()
                    break
        LogDebug("end to run [%s] at episode [%d].", self.pid, i_episode)
        return self.pid


    def send_sample(self, sample_info):
        ret = self.redis.push_sample(sample_info)
        #LogDebug("pid[%s] send_sample ret[%s]", self.pid, ret)
    
    def wait_to_update_latest_model(self):
        self.model_name = "yh_cartpole_model"
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