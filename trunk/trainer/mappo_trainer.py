import os
import torch 
import numpy as np

from time import time, gmtime, strftime
from torch.utils.tensorboard import SummaryWriter
from runner import create_runner
from config import global_args as g_args
from learner import create_learner
import setproctitle
from buffer import ExpertBuffer

class MAPPOTrainer:
    def __init__(self):
        self.opponent = ""
        self.train_idx = 0  # 第几次训练
        self.logger = None
        self.learner = create_learner(g_args("learner"))
        self.runner = create_runner("multi_thread_runner")
        self.name = "default"
        self.thread_num = g_args("thread_num")
        self.num_env_steps = g_args("num_env_steps")
        self.episode_length = g_args("episode_length")
        self.use_linear_lr_decay = g_args("use_linear_lr_decay")
        self.save_interval = g_args("save_interval")
        self.eval_interval = g_args("eval_interval")
        self.log_interval = g_args("log_interval")
        self.use_eval =  g_args("use_eval")

    def set_opponent(self, opponent):
        self.opponent = opponent
        self.runner.setup(self.learner, opponent)

    def set_name(self, name):
        self.name = name
        
    def init_logger(self):
        #精确到小时就好，一小时都没训练到的日志，没必要保存
        current_time = strftime("%Y-%m-%d-%H", gmtime())
        #画图
        writer_path = f"logs/mappo/{current_time}/{self.name}_{self.train_idx}"
        setproctitle.setproctitle(writer_path)
        if not os.path.exists(writer_path): 
            os.makedirs(writer_path)
        print(writer_path)
        self.logger = SummaryWriter(log_dir=writer_path)

    def train(self):
        self.train_idx += 1
        self.init_logger()
        episodes = self.num_env_steps // self.episode_length // self.thread_num
        start = time()
        self.runner.clear_game_results()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.learner.lr_decay(episode, episodes)
            # 获得训练数据
            buffer = self.runner.run()
            # 计算并更新网络
            train_infos = self.learner.learn(buffer)
            # 记录日志
            if self.use_eval and episode % self.eval_interval == 0:
                info = self.runner.eval()
                self.log_eval(info, episode)
            if episode % self.log_interval == 0:
                self.log_info(episode, start, train_infos)
        # 训练结束后，直接进行一次评估
        result = self.runner.eval()
        self.log_eval(result, episode)
        # 训练结束后再进行一次保存
        self.save()
    
    def get_game_results(self):
        return self.runner.get_game_results()

    def get_steps(self):
        return self.num_env_steps

    def save(self):
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        self.learner.save_models(f"./models/mappo/{current_time}/{self.name}")

    def load_from(self, model_name):
        #TODO
        self.learner.load_models(f"./models/mappo/{model_name}")

    def log_info(self, episode, start, train_infos):
        now = time()
        episodes = self.num_env_steps // self.episode_length // self.thread_num
        total_steps = (episode + 1) * self.episode_length * self.thread_num
        print(f"{self.name} time_used {now - start} episodes {episode + 1}/{episodes} timesteps {total_steps}/{self.num_env_steps} .\n")
        for k, v in train_infos.items():
            self.logger.add_scalar(k, v, total_steps)

    def log_eval(self, env_infos, episode):
        total_num_steps = (episode + 1) * self.episode_length * self.thread_num
        for k, v in env_infos.items():
            if len(v)>0:
                self.logger.add_scalar(k, np.mean(v), total_num_steps)

    def get_model(self):
        return self.learner.save_dict()