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
from models import Pi_net, V_net, ActorNet, CriticNet

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
        self.env = gym.make('Pendulum-v1')
        n_states = self.env.observation_space.shape[0]

        self.bound = self.env.action_space.high[0]
        self.actor_model = ActorNet(n_states, self.bound).to(DEVICE)
        self.critic_model = CriticNet(n_states).to(DEVICE)
        self.actor_old_model = ActorNet(n_states, self.bound).to(DEVICE)
        
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr)

        self.save_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/"
        self.learner_log_path = os.path.dirname(os.path.abspath(__file__)) + "../../logs/"
        self.memory_pool = CreateMemPool({"host": '172.18.232.25', "password": 'Lab504redis', "port": 6379, "test":False}) 
        self.loss_fn = nn.MSELoss()

        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir="/home/yinhan/yh_acadamy_football/logs/Pendulum_new")   

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).reshape(-1, 1).to(DEVICE)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)                           # torch.Size([batch, 1])
        loss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states).to(DEVICE)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step() 
    
    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states).to(DEVICE)
        v = self.critic_model(states)                             # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach() 

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())        # 首先更新旧模型
        advantage = self.cal_adv(states, targets)

        for i in range(self.K_epochs):                      # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.K_epochs):                      # 更新多次
            self.critic_learn(states, targets)
    
    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_).to(DEVICE)
        target = self.critic_model(s_).detach()                 # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)                   # torch.Size([batch])

        return target_list

    def start(self):
        step = 0
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
                    s, a, r, s_, done, l, num = self.preprocess(episode_data)
                    i_episode += num
                    step += l
                    LogDebug("Learner step:[%d]", step)
                    targets = self.discount_reward(r, s_[-1])
                    self.update(s, a, targets)
                
                    self.save_model(0, "yh_Pendulum_model")
                    episode_data.clear()
                    break

            if  step % 200 == 0:        
                avg_length, r = self.test()
                self.writer.add_scalars("sampled/step", {"reward": step}, step)
        #self.writer.add_scalar("empty_goal_entropy_loss", scalar_value = entropy, global_step = i_episode)
                LogDebug("Reward [%d] \t episode:[%lf]", r,  i_episode)

        end_time = time.time()
        LogDebug("Train of learner has finished after [%f]s.", end_time - self.start_time)

    def push_model(self, env_id, model_name, model_obj):
        self.memory_pool.push_model(env_id, model_name, model_obj)

    def preprocess(self, collect_data):
        max_step = self.args.update_interval // self.args.num_workers

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []
        next_obs_batch = []
        for i in range(self.args.num_workers):
            data = collect_data[i]
            for step in range(max_step):
                obs_batch.append(data[step]["obs"])
                actions_batch.append(data[step]["action"])
                rewards_batch.append(data[step]["reward"])
                next_obs_batch.append(data[step]["next_obs"])
                dones_batch.append(data[step]["done"])
        l = len(obs_batch)
        episode = dones_batch.count(torch.tensor(1))
        s = np.array(obs_batch)
        a = np.array(actions_batch)
        r = np.array(rewards_batch)
        s_ = np.array(next_obs_batch)
        done = np.array(dones_batch)
        return s, a, r, s_, done, l, episode


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

        torch.save(self.actor_model.state_dict(), path+"Actor.pth")
        # 向数据库推送一个
        net = copy.deepcopy(self.actor_model)
        params = net.state_dict()
        model_obj = Model(params)
        self.push_model(env_id, model_name, model_obj)
    

    def load_model(self, env_id, name):
        path = self.save_model_path + env_id + "/" + name + "/"
        policy_model = torch.load(path + "Actor.pth")

        self.actor_model.load_state_dict(policy_model)
        # 向数据库推送一个
        params = self.actor_model.state_dict()
        model_obj = Model(params)
        self.push_model(env_id, name, model_obj)
    
    def write_log(self, tag, global_step, **dict_value):
        for key in dict_value:
            self.writer.add_scalar(tag=tag+'/'+key, scalar_value=dict_value[key], global_step=global_step)


    def update_new_model(self, path):
        self.actor_model.load_state_dict(torch.load(path+"Actor.pth"))

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(DEVICE)
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return np.clip(action.cpu().numpy(), -self.bound, self.bound)

    def test(self):
        obs = self.env.reset()
        total_reward = 0
        length = 0
        for i in range(200):
            obs = self.env.reset()
        return length, total_reward