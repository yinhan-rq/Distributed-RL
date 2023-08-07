import os
import time
import torch
import numpy as np
import copy
from log import Log, LogDebug, LogErr, LogExc
from torch.distributions import Categorical
from torch import optim, nn
from time import sleep

import gym
import torch.nn.functional as F
from torch import nn, optim, Tensor
from ..mempool import CreateMemPool, Model
from models import Policy

from torch.utils.tensorboard import SummaryWriter

torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print("used device",DEVICE)



class Learner(object):
    """
    Learner 负责从mempool 拉取采样数据，向redis写入模型
    """
    def __init__(self, args):
        self.args = args
        self.device = DEVICE
        self.gamma = args.gamma
        self.epsilon = args.eps
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.ent_coef
        self.clip_param = args.clip
        self.gae_lambda = args.tau
        self.batch_size = args.batch_size
        self.lr = self.args.lr
        self.K_epochs = self.args.epoch
        self.betas = self.args.betas
        self.env = gym.make('CartPole-v1')
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(DEVICE)
        self.optim = optim.Adam(self.policy.parameters(), lr=args.lr, betas=self.betas)

        self.save_vedio_path = os.path.dirname(os.path.abspath(__file__)) + "/vedio/"
        self.save_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/"
        self.learner_log_path = os.path.dirname(os.path.abspath(__file__)) + "../../logs/"
        self.memory_pool = CreateMemPool({"host": '172.18.232.25', "password": 'Lab504redis', "port": 6379, "test":False}) 
        self.loss_fn = nn.MSELoss()

        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir="/home/yinhan/yh_acadamy_football/logs/cartpole")    

        if not os.path.exists(self.save_vedio_path):
            os.makedirs(self.save_vedio_path)

    def update_network(self, obs_batch, reward_batch, actions_batch, prob_batch, done_batch):
        returns = self.cal_returns(np.array(reward_batch), done_batch)
        returns = torch.tensor(returns).float().to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        obs_batch = torch.Tensor(np.stack(obs_batch)).to(DEVICE).detach()
        actions_batch = torch.Tensor(actions_batch).to(DEVICE).detach()
        prob_batch = torch.stack(prob_batch).to(DEVICE).detach()

        for _ in range(self.K_epochs):
            data_generator = self.feed_forward_generator(obs_batch, reward_batch, actions_batch, prob_batch, returns)

            for sample in data_generator: 
                split_returns = sample["returns_batch"]
                split_old_states = sample["obs_batch"]
                split_old_actions = sample["actions_batch"]
                split_old_logprobs = sample["old_prob_batch"]

                log_probs, state_values = self.policy.evaluate(split_old_states, split_old_actions)

                # adv normalize
                adv = split_returns - state_values.detach()

                imp_weights = torch.exp(log_probs - split_old_logprobs.detach())

                surr1 = imp_weights * adv
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = self.loss_fn(state_values, split_returns)

                total_loss = policy_loss + self.args.vloss_coef * value_loss

                self.optim.zero_grad()
                total_loss.mean().backward()
                self.optim.step()
        


    def start(self):
        step = 0
        sample = 0
        i_episode = 0
        #self.load_model(str(0), "yh_academy_football_model")
        get_num = 0
        episode_data = []
        for i_episode in range(self.args.max_episode):
            while True:
            # 学习率调整
                episode_data.append(self.pull_sample())
                get_num += 1
                if get_num == self.args.num_workers:
                    LogDebug("Learner has got a batch data.")
                    get_num = 0
                    obs_batch, reward_batch, action_batch, old_prob_batch, done_batch = self.preprocess(episode_data)
                    #step += self.args.nsteps
                    #sample += self.args.nsteps
                    i_episode += done_batch.count(1)
                    step += len(obs_batch)
                    LogDebug("Learner step:[%d]", step)
                    self.update_network(obs_batch, reward_batch, action_batch, old_prob_batch, done_batch)
                
                    self.save_model(0, "yh_cartpole_model")
                    episode_data.clear()
                    break

            if  step % 400 == 0:        
                avg_length, r = self.test()
                self.writer.add_scalars("reward/episode", {"reward": r}, i_episode)
                self.writer.add_scalars("reward/step", {"reward": r}, step)
        #self.writer.add_scalar("empty_goal_entropy_loss", scalar_value = entropy, global_step = i_episode)
                #LogDebug("Reward [%d] \t episode:[%lf]", r,  i_episode)

        end_time = time.time()
        LogDebug("Train of learner has finished after [%f]s.", end_time - self.start_time)

    def push_model(self, env_id, model_name, model_obj):
        self.memory_pool.push_model(env_id, model_name, model_obj)

    def preprocess(self, collect_data):
        max_step = self.args.update_interval // self.args.num_workers

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        probs_batch = []
        dones_batch = []
        for i in range(self.args.num_workers):
            data = collect_data[i]
            for step in range(max_step):
                obs_batch.append(data[step]["obs"])
                actions_batch.append(data[step]["action"])
                rewards_batch.append(data[step]["reward"])
                probs_batch.append(data[step]["old_probs"])
                dones_batch.append(data[step]["done"])

        return obs_batch, rewards_batch, actions_batch, probs_batch, dones_batch

    def cal_returns(self, rewards, dones):
        returns = np.zeros_like(rewards)
        discount_reward = 0
        for step in reversed(range(rewards.shape[0])):
            if dones[step]:
                discount_reward = 0
            discount_reward = rewards[step] + self.gamma * discount_reward
            returns[step] = discount_reward
        return returns

    def pull_sample(self): # 获取采样数据
        sample = self.memory_pool.pull_sample()
        return sample
    
    
    def save_model(self, env_id, model_name=None):
        if not model_name:
            model_name = time.strftime("%Y_%m_%d_%H_%M")
        path = self.save_model_path + str(env_id) + "/" + model_name + "/"

        # 本地保存一个
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.policy.state_dict(), path+"policy.pth")
        # 向数据库推送一个
        net = copy.deepcopy(self.policy)
        params = net.cpu().state_dict()
        model_obj = Model(params)
        self.push_model(env_id, model_name, model_obj)
    

    def load_model(self, env_id, name):
        path = self.save_model_path + env_id + "/" + name + "/"
        policy_model = torch.load(path + "policy.pth")

        self.policy.load_state_dict(policy_model)
        # 向数据库推送一个
        params = self.policy.state_dict()
        model_obj = Model(params)
        self.push_model(env_id, name, model_obj)
    
    def write_log(self, tag, global_step, **dict_value):
        for key in dict_value:
            self.writer.add_scalar(tag=tag+'/'+key, scalar_value=dict_value[key], global_step=global_step)


    def update_new_model(self, path):
        self.policy.load_state_dict(torch.load(path+"policy.pth"))

    
    def feed_forward_generator(self, obs, rewards, actions, probs, returns):
        rewards = torch.tensor(rewards).to(DEVICE)
        rand = torch.randperm(rewards.shape[0]).numpy()
        num_batch = rewards.shape[0] // self.args.batch_size
        sampler = [rand[i * self.args.batch_size : (i + 1) * self.args.batch_size] for i in range(num_batch)]
        for indice in sampler:
            obs_batch = obs[indice]
            rewards_batch = rewards[indice]
            actions_batch = actions[indice]
            old_prob_batch = probs[indice]
            returns_batch = returns[indice]

            sample = {}
            sample["obs_batch"] =  obs_batch
            sample["rewards_batch"] = rewards_batch
            sample["actions_batch"] = actions_batch
            sample["old_prob_batch"] = old_prob_batch
            sample["returns_batch"] = returns_batch
            yield sample
    
    def test(self):
        obs = self.env.reset()
        total_reward = 0
        length = 0
        while True:
            with torch.no_grad():
                action, log_prob = self.policy.act(torch.tensor(obs).to(DEVICE))
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            length += 1
            if done:
                break
        return length, total_reward