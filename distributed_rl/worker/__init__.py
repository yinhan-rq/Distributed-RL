# -*- coding: utf-8 -*-
import os
from torch import optim, nn 
from time import sleep
import traceback
import torch
import numpy as np
from copy import copy
#from train import env as multi_scenario
import distributed_rl.mempool as MEMPOOL
import gfootball.env as football_env
from log import LogDebug, LogErr, LogExc
from models import ActorCritic as net
from encoder import FeatureEncoder
from Rewarder import calc_reward
#import ray

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else'cpu') 
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
        self.push_interval = args.update_interval // self.args.num_workers
        self.env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='simple115v2',
                    rewards = 'scoring',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    logdir="./trace"
                    )

        self.policy = net(self.env.observation_space.shape[0], self.env.action_space.n, 64)#.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.loss_fn = nn.MSELoss()
        redis_config ={}
        redis_config["host"] = self.args.redis_ip
        redis_config["password"] = self.args.redis_pass_word
        redis_config["port"] = self.args.redis_port
        self.redis = MEMPOOL.CreateMemPool(redis_config)
        print("create worker "+ str(self.pid))
        

    def run(self):
        LogDebug("begin to run [%s]", self.pid)
        num_updates = self.args.max_episode // (self.args.num_workers)
        timestep = 0
        episode = {}
        t = 0
        obs = self.env.reset()
        for i_episode in range(1, num_updates + 1):
            t = i_episode
            for step in range(self.args.nsteps):
                trajectory = {}
                action, log_prob = self.policy.act(torch.from_numpy(obs).float())
                obs = torch.from_numpy(obs).float().to(device)
                trajectory["obs"] = obs
                trajectory["action"] = torch.tensor(action)
                trajectory["old_probs"] = log_prob

                obs, reward, done, _ = self.env.step(action)

                trajectory["reward"] = reward
                trajectory["done"] = done

                episode[timestep] = trajectory
                timestep += 1

                if timestep >= self.push_interval:
                    self.send_sample(episode)
                    episode = {}
                    timestep = 0
                    self.wait_to_update_latest_model()

                if done:
                    obs = self.env.reset()
                    break
        LogDebug("end to run [%s] and episode = [%d]", self.pid, t)
        return self.pid


    def send_sample(self, sample_info):
        ret = self.redis.push_sample(sample_info)
        LogDebug("pid[%s] send_sample ret[%s]", self.pid, ret)
    
    def wait_to_update_latest_model(self):
        self.model_name = "yh_academy_football_model"
        while True:
            robj = self.redis.pull_model(0, self.model_name)
            if not robj:
                LogDebug("Wait fot new model.")
                sleep(5)
            else:
                break
        model_data = robj.data
        self.policy.load_state_dict(model_data)
        LogDebug(f"worker[{self.args.env_name}] model[{self.model_name}] update finish")


class Worker_grad(object):
    """
    Worker 从redis拉取模型，采样数据，更新，把梯度写回到mempool
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
        self.push_interval = args.update_interval // self.args.num_workers
        self.env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='simple115v2',
                    rewards = 'scoring',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    logdir="./trace"
                    )

        self.policy = net(self.env.observation_space.shape[0], self.env.action_space.n, 64).to(device)
        #self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.loss_fn = nn.MSELoss()
        redis_config ={}
        redis_config["host"] = self.args.redis_ip
        redis_config["password"] = self.args.redis_pass_word
        redis_config["port"] = self.args.redis_port
        self.redis = MEMPOOL.CreateMemPool(redis_config)
        print("create worker "+ str(self.pid))
    
    def split(self, actions, batch_size):
        split_res = []
        length = len(actions)
        if batch_size is None:
            split_res.append((0, length-1))
            return split_res
        for idx in range(0, length, batch_size):
            if idx + batch_size < length:
                split_res.append((idx, idx + batch_size))
            else:
                split_res.append((idx, length))
        return split_res
    
    def run(self):
        LogDebug("begin to run [%s]", self.pid)
        num_updates = self.args.max_episode // (self.args.num_workers)
        timestep = 0
        obs_batch, reward_batch, actions_batch, dones_batch, probs_batch = [], [], [], [], []

        for i_episode in range(1, self.args.max_episode + 1):

            obs = self.env.reset()
            for t in range(self.args.nsteps):
                timestep += 1
                action, log_prob = self.policy.act(obs)
                obs_tensor = torch.tensor(obs).to(device)
                obs_batch.append(obs_tensor)     
                actions_batch.append(torch.tensor(action))    
                probs_batch.append(log_prob)

                obs, reward, done, _ = self.env.step(action)

                reward_batch.append(reward)
                dones_batch.append(done)

                if timestep % self.push_interval == 0:
                    self.update_network(obs_batch, actions_batch, reward_batch, dones_batch, probs_batch)
                    timestep = 0
                    obs_batch.clear()
                    reward_batch.clear()
                    actions_batch.clear()
                    dones_batch.clear() 
                    probs_batch.clear()

                if done:
                    break
        return self.pid
    
    def update_network(self, obs_batch, actions_batch, reward_batch, done_batch, prob_batch):
        returns = self.cal_returns(np.array(reward_batch), done_batch)
        returns = torch.tensor(returns).float().to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        obs_batch = torch.stack(obs_batch).to(device).detach()
        actions_batch = torch.stack(actions_batch).to(device).detach()
        prob_batch = torch.stack(prob_batch).to(device).detach()

        split_res = self.split(actions_batch, self.args.batch_size)

        for _ in range(self.K_epochs):
            for split_idx in split_res:
                split_old_states = obs_batch[split_idx[0]:split_idx[1]]
                split_old_actions = actions_batch[split_idx[0]:split_idx[1]]
                split_old_logprobs = prob_batch[split_idx[0]:split_idx[1]]
                split_returns = returns[split_idx[0]:split_idx[1]]

                log_probs, state_values, dist_entropy = self.policy.evaluate(split_old_states, split_old_actions)
                # adv normalize
                adv = split_returns - state_values.detach()

                imp_weights = torch.exp(log_probs - split_old_logprobs.detach())

                surr1 = imp_weights * adv
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip, 1.0 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = self.loss_fn(state_values, split_returns)

                #actor_loss = -(log_probs * adv.detach()).mean()

                #critic_loss = self.loss_fn(state_values, split_returns)

                total_loss = policy_loss + self.args.vloss_coef * value_loss - self.args.ent_coef * dist_entropy
                total_loss.mean().backward()
                self.send_grad(self.policy, self.args.env_name)
                self.wait_to_update_latest_model()

    def send_grad(self, policy, model_name):
        self.redis.push_grad(policy, model_name)
    
    def wait_to_update_latest_model(self):
        self.model_name = "grad_model"
        while True:
            robj = self.redis.pull_model(0, self.model_name)
            #print(self.redis.get_grad_list_len(self.args.env_name))
            if (not robj) or self.redis.get_grad_list_len(self.args.env_name):
                LogDebug("Wait fot new model.")
                sleep(5)
            else:
                break
        model_data = robj.data
        self.policy.load_state_dict(model_data)
        LogDebug(f"worker[{self.args.env_name}] model[{self.model_name}] update finish")
    
    def cal_returns(self, rewards, dones):
        returns = np.zeros_like(rewards)
        discount_reward = 0
        for step in reversed(range(rewards.shape[0])):
            if dones[step]:
                discount_reward = 0
            discount_reward = rewards[step] + self.gamma * discount_reward
            returns[step] = discount_reward
        return returns




class Worker_multi(object):
    '''
        Worker for 5_vs_5 and 11_vs_11
    '''
    def __init__(self, args):
        self.pid = os.getpid()
        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.clip = args.clip
        self.K_epochs = args.epoch
        self.betas = args.betas
        self.ent_coef = args.ent_coef
        self.push_interval = args.update_interval // 4 #self.args.num_workers
        self.env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='raw',
                    rewards = 'scoring, checkpoints',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    logdir="./trace"
                    )
        obs = self.env.reset()
        self.fe = FeatureEncoder()
        obs = self.fe.Encoder(obs[0])

        self.policy = net(self.fe.concatenate_obs(obs).shape[0], self.env.action_space.n, 64)#.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.loss_fn = nn.MSELoss()
        redis_config ={}
        redis_config["host"] = self.args.redis_ip
        redis_config["password"] = self.args.redis_pass_word
        redis_config["port"] = self.args.redis_port
        self.redis = MEMPOOL.CreateMemPool(redis_config)
        print("create worker "+ str(self.pid))
        

    def run(self):
        LogDebug("begin to run [%s]", self.pid)
        num_updates = self.args.max_episode // 4#(self.args.num_workers)
        timestep = 0
        episode = {}
        t = 0
        obs = self.env.reset()
        pre_obs = obs
        for i_episode in range(1, num_updates + 1):
            t = i_episode
            for step in range(2000):
                trajectory = {}
                np_obs = self.fe.Encoder(obs[0])
                np_obs = self.fe.concatenate_obs(np_obs) 
                action, log_prob = self.policy.act(torch.from_numpy(np_obs).float())
                np_obs = torch.from_numpy(np_obs).float().to(device)
                trajectory["obs"] = np_obs
                trajectory["action"] = torch.tensor(action)
                trajectory["old_probs"] = log_prob

                obs, reward, done, _ = self.env.step(action)
                reward = calc_reward(reward, pre_obs[0], obs[0])
                trajectory["reward"] = reward
                trajectory["done"] = done

                pre_obs = obs

                episode[timestep] = trajectory
                timestep += 1

                if timestep == self.push_interval:
                    self.send_sample(episode)
                    episode = {}
                    timestep = 0
                    self.wait_to_update_latest_model()

                if done:
                    obs = self.env.reset()
                    pre_obs = obs
                    break
        LogDebug("end to run [%s] and episode = [%d]", self.pid, t)
        return self.pid


    def send_sample(self, sample_info):
        ret = self.redis.push_sample(sample_info)
        LogDebug("pid[%s] send_sample ret[%s]", self.pid, ret)


    
    def wait_to_update_latest_model(self):
        self.model_name = "yh_academy_football_5v5_model"
        while True:
            robj = self.redis.pull_model(0, self.model_name)
            if (not robj) or self.redis.get_sample_list_len():
                LogDebug("Wait fot new model.")
                sleep(15)
            else:
                break
        model_data = robj.data
        self.policy.load_state_dict(model_data)
        LogDebug(f"worker[{self.args.env_name}] model[{self.model_name}] update finish")
    


