from hashlib import new
import os
import time
import torch
import numpy as np
import copy
from log import Log, LogDebug, LogErr, LogExc
from torch.distributions import Categorical
from torch import optim, nn
from time import sleep

from ..mempool import CreateMemPool, Model
from models import ActorCritic as net
import gfootball.env as football_env
from encoder import FeatureEncoder
from Rewarder import calc_reward

from torch.utils.tensorboard import SummaryWriter

torch.cuda.manual_seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        self.log_interval = self.args.log_interval
        self.test_env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='simple115v2',
                    rewards = 'scoring',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    write_video=False,
                    render=False
                    )
        
        self.policy = net(self.test_env.observation_space.shape[0], self.test_env.action_space.n, 64).to(DEVICE)

        self.save_vedio_path = os.path.dirname(os.path.abspath(__file__)) + "/vedio/"
        self.save_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/"
        self.learner_log_path = os.path.dirname(os.path.abspath(__file__)) + "../../logs/"
        self.memory_pool = CreateMemPool({"host": '172.18.232.25', "password": 'Lab504redis', "port": 6379, "test":False}) 
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), self.lr, eps=self.epsilon)
        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir="/home/yinhan/yh_acadamy_football/logs/3_vs_1_with_keeper_new")    

        if not os.path.exists(self.save_vedio_path):
            os.makedirs(self.save_vedio_path)

    def update_network(self, obs_batch, reward_batch, actions_batch, prob_batch, done_batch):
        returns = self.cal_returns(np.array(reward_batch), done_batch)
        returns = torch.tensor(returns).float().to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        obs_batch = torch.stack(obs_batch).to(DEVICE).detach()
        actions_batch = torch.stack(actions_batch).to(DEVICE).detach()
        prob_batch = torch.stack(prob_batch).to(DEVICE).detach()

        ent_loss = 0
        for _ in range(self.K_epochs):
            data_generator = self.feed_forward_generator(obs_batch, reward_batch, actions_batch, prob_batch, returns)

            for sample in data_generator: 
                split_returns = sample["returns_batch"]
                split_old_states = sample["obs_batch"]
                split_old_actions = sample["actions_batch"]
                split_old_logprobs = sample["old_prob_batch"]

                log_probs, state_values, dist_entropy = self.policy.evaluate(split_old_states, split_old_actions)
                ent_loss = torch.sum(dist_entropy) / dist_entropy.shape[0]
                # adv normalize
                adv = split_returns - state_values.detach()

                imp_weights = torch.exp(log_probs - split_old_logprobs.detach())

                surr1 = imp_weights * adv
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = self.loss_fn(state_values, split_returns)

                total_loss = policy_loss + self.args.vloss_coef * value_loss - self.args.ent_coef * dist_entropy

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                self.optimizer.step()
        return ent_loss
        


    def start(self):
        i_episode = 0
        #self.load_model(str(0), "yh_academy_football_model")
        get_num = 0
        episode_data = []
        while True:
            # 学习率调整
            episode_data.append(self.pull_sample())
            get_num += 1
            if get_num == self.args.num_workers:
                LogDebug("Learner has got a batch data.")
                get_num = 0
                obs_batch, reward_batch, action_batch, old_prob_batch, done_batch = self.preprocess(episode_data)
                i_episode += done_batch.count(1)
                LogDebug("Learner episode:[%d]", i_episode)
                ent_loss = self.update_network(obs_batch, reward_batch, action_batch, old_prob_batch, done_batch)
                
                #self.push_model(0, "yh_academy_football_model", Model(self.policy.state_dict()))
                self.save_model(0, "yh_academy_football_model")
                episode_data.clear()
            
            if i_episode % 10 == 0:
                entropy, avg_reward = self.test()
                self.writer.add_scalars("reward/episode", {"reward of parallel": avg_reward}, i_episode)
                #self.writer.add_scalar("empty_goal_entropy_loss", scalar_value = entropy, global_step = i_episode)
                LogDebug("Episode [%d] \t reward:[%lf]", i_episode,  avg_reward)
            
            if i_episode >= self.args.max_episode:
                break

            
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
        obs = self.test_env.reset()
        total_reward = 0
        times = 0
        for _ in range(1000):
            with torch.no_grad():
                action, pi = self.policy.act(torch.tensor(obs).to(DEVICE))
                _, _, entropy = self.policy.evaluate(torch.tensor(obs).to(DEVICE), torch.tensor(action).to(DEVICE))
            obs, reward, done, _ = self.test_env.step(action)
            total_reward += reward
            if done:
                times += 1
                obs = self.test_env.reset()
        return entropy, total_reward / times
    


class Learner_multi(object):
    '''
        负责5_vs_5 and 11_vs_11的Learner
    '''
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
        self.log_interval = self.args.log_interval
        self.test_env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='raw',
                    rewards = 'scoring,checkpoints',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    write_video=False,
                    render=False
                    )
        obs = self.test_env.reset()
        self.fe = FeatureEncoder()
        obs = self.fe.Encoder(obs[0])
        self.policy = net(self.fe.concatenate_obs(obs).shape[0], self.test_env.action_space.n, 64).to(DEVICE)

        self.save_vedio_path = os.path.dirname(os.path.abspath(__file__)) + "/vedio/"
        self.save_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/"
        self.learner_log_path = os.path.dirname(os.path.abspath(__file__)) + "../../logs/"
        self.memory_pool = CreateMemPool({"host": '172.18.232.25', "password": 'Lab504redis', "port": 6379, "test":False}) 
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), self.lr, eps=self.epsilon)
        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir="/home/yinhan/yh_acadamy_football/logs/5_vs_5")    

        if not os.path.exists(self.save_vedio_path):
            os.makedirs(self.save_vedio_path)

    def update_network(self, obs_batch, reward_batch, actions_batch, prob_batch, done_batch):
        returns = self.cal_returns(np.array(reward_batch), done_batch)
        returns = torch.tensor(returns).float().to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        obs_batch = torch.stack(obs_batch).to(DEVICE).detach()
        actions_batch = torch.stack(actions_batch).to(DEVICE).detach()
        prob_batch = torch.stack(prob_batch).to(DEVICE).detach()

        ent_loss = 0
        for _ in range(self.K_epochs):
            data_generator = self.feed_forward_generator(obs_batch, reward_batch, actions_batch, prob_batch, returns)

            for sample in data_generator: 
                split_returns = sample["returns_batch"]
                split_old_states = sample["obs_batch"]
                split_old_actions = sample["actions_batch"]
                split_old_logprobs = sample["old_prob_batch"]

                log_probs, state_values, dist_entropy = self.policy.evaluate(split_old_states, split_old_actions)
                ent_loss = torch.sum(dist_entropy) / dist_entropy.shape[0]
                # adv normalize
                adv = split_returns - state_values.detach()

                imp_weights = torch.exp(log_probs - split_old_logprobs.detach())

                surr1 = imp_weights * adv
                surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = self.loss_fn(state_values, split_returns)

                total_loss = policy_loss + self.args.vloss_coef * value_loss - self.args.ent_coef * dist_entropy

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                self.optimizer.step()
        return ent_loss
        


    def start(self):
        i_episode = 0
        #self.load_model(str(0), "yh_academy_football_model")
        get_num = 0
        episode_data = []
        while True:
            # 学习率调整
            episode_data.append(self.pull_sample())
            get_num += 1
            LogDebug("Learner has got [%d] sample.", get_num)
            if get_num == 4:
                LogDebug("Learner has got a batch data.")
                get_num = 0
                obs_batch, reward_batch, action_batch, old_prob_batch, done_batch = self.preprocess(episode_data)
                i_episode += done_batch.count(1)
                LogDebug("Learner episode:[%d]", i_episode)
                ent_loss = self.update_network(obs_batch, reward_batch, action_batch, old_prob_batch, done_batch)
                
                #self.push_model(0, "yh_academy_football_model", Model(self.policy.state_dict()))
                self.save_model(0, "yh_academy_football_5v5_model")
                episode_data.clear()
            
            if i_episode % 10 == 0 and i_episode:
                avg_reward = self.test()
                self.writer.add_scalars("reward/episode", {"reward of parallel": avg_reward}, i_episode)
                #self.writer.add_scalar("empty_goal_entropy_loss", scalar_value = entropy, global_step = i_episode)
                #LogDebug("Episode [%d] \t avg length: [%d]\t reward:[%lf]", i_episode, avg_length, avg_reward)
            
            if i_episode >= self.args.max_episode:
                break

            
        end_time = time.time()

        LogDebug("Train of learner has finished after [%f]s.", end_time - self.start_time)

    def push_model(self, env_id, model_name, model_obj):
        self.memory_pool.push_model(env_id, model_name, model_obj)

    def preprocess(self, collect_data):
        max_step = self.args.update_interval // 4

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        probs_batch = []
        dones_batch = []
        for i in range(4):
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
        obs = self.test_env.reset()
        total_reward = 0
        times = 0
        for _ in range(9000+1):
            with torch.no_grad():
                np_obs = self.fe.Encoder(obs[0])
                np_obs = self.fe.concatenate_obs(np_obs)
                action, pi = self.policy.act(torch.tensor(np_obs).to(DEVICE))
                #_, _, entropy = self.policy.evaluate(torch.tensor(obs).to(DEVICE), torch.tensor(action).to(DEVICE))
            obs, reward, done, _ = self.test_env.step(action)
            total_reward += reward
            if done:
                times += 1
                obs = self.test_env.reset()
        return total_reward / times


class learner_grad(object):
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
        self.log_interval = self.args.log_interval
        self.test_env = football_env.create_environment(
                    env_name=args.env_name,
                    stacked=False,
                    representation='simple115v2',
                    rewards = 'scoring',
                    write_goal_dumps=False,
                    write_full_episode_dumps=False,
                    write_video=False,
                    render=False
                    )
        
        self.policy = net(self.test_env.observation_space.shape[0], self.test_env.action_space.n, 64).to(DEVICE)

        self.save_vedio_path = os.path.dirname(os.path.abspath(__file__)) + "/vedio/"
        self.save_model_path = os.path.dirname(os.path.abspath(__file__)) + "/model/"
        self.learner_log_path = os.path.dirname(os.path.abspath(__file__)) + "../../logs/"
        self.memory_pool = CreateMemPool({"host": '172.18.232.25', "password": 'Lab504redis', "port": 6379, "test":False}) 
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), self.lr, eps=self.epsilon)
        self.start_time = time.time()
        self.writer = SummaryWriter(log_dir="/home/yinhan/yh_acadamy_football/logs/3_vs_1_new")    

        if not os.path.exists(self.save_vedio_path):
            os.makedirs(self.save_vedio_path)


    def start(self):
        i_episode = 0
        #self.load_model(str(0), "yh_academy_football_model")
        get_num = 0
        while True:
            # 学习率调整
            worker_net = self.pull_grad(self.args.env_name)
            get_num += 1
            self.optimizer.zero_grad()
            self.update_network(worker_net)
            i_episode += (self.args.batch_size) / (2 * self.args.nsteps * self.args.epoch)
            if get_num == self.args.num_workers:
                LogDebug("Learner has got a batch grad.")
                get_num = 0
                #self.push_model(0, "yh_academy_football_model", Model(self.policy.state_dict()))
                self.save_model(0, "grad_model")
            
            if i_episode % 10 == 0:
                entropy, avg_reward = self.test()
                self.writer.add_scalars("reward/episode", {"reward of parallel": avg_reward}, i_episode)
                #self.writer.add_scalar("empty_goal_entropy_loss", scalar_value = entropy, global_step = i_episode)
                #LogDebug("Episode [%d] \t avg length: [%d]\t reward:[%lf]", i_episode, avg_length, avg_reward)
            
            if i_episode >= self.args.max_episode:
                break

            
        end_time = time.time()

        LogDebug("Train of learner has finished after [%f]s.", end_time - self.start_time)
    
    def update_network(self, worker_net):
        for lp, gp in zip(worker_net.parameters(), self.policy.parameters()):
            gp.grad = lp.grad
        self.optimizer.step()

    def push_model(self, env_id, model_name, model_obj):
        self.memory_pool.push_model(env_id, model_name, model_obj)
    
    def pull_grad(self, model_name):
        grad = self.memory_pool.pull_grad(model_name)
        return grad
    
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
    
    def test(self):
        obs = self.test_env.reset()
        total_reward = 0
        times = 0
        for _ in range(1000):
            with torch.no_grad():
                action, pi = self.policy.act(obs)
                _, _, entropy = self.policy.evaluate(torch.tensor(obs).to(DEVICE), torch.tensor(action).to(DEVICE))
            obs, reward, done, _ = self.test_env.step(action)
            total_reward += reward
            if done:
                times += 1
                obs = self.test_env.reset()
        return entropy, total_reward / times