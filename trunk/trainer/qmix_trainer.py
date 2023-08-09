
from asyncore import write
from runner import create_runner
from config import global_args as g_args
from learner import create_learner
from time import gmtime, strftime
import os
from torch.utils.tensorboard import SummaryWriter

class QMixTrainer:

    def __init__(self) -> None:
        self.runner = create_runner("episode_runner")
        self.learner = create_learner("qmix_learner")
        self.max_timestep = g_args("time_step")
        self.batch_size = g_args("batch_size")
        self.logger = None
        self.name = "default"
        self.train_idx = 0
        self.opponent = "random"
        self.init_logger()
        print(self.max_timestep, self.batch_size)

    def set_name(self, name):
        self.name = name

    def set_opponent(self, opponent):
        self.opponent = opponent

    def train(self):
        self.train_idx += 1
        self.runner.setup(self.learner, self.opponent)
        episode = 0
        while self.runner.timestep() < self.max_timestep:
            print("episode:", episode)
            self.runner.run() # 一局游戏 
            if self.runner.can_sample():  # 训练
                batch = self.runner.sample()
                loss = self.runner.agent.train(batch)  
                self.logger.add_scalar("loss", loss, episode)
            if episode % g_args("test_frequency")==0: 
                my_avg_reward,opponent_avg_reward,win_rate = self.runner.test( g_args("test_episodes"))
                self.logger.add_scalar("my_avg_reward",my_avg_reward,episode)
                self.logger.add_scalar("opponent_avg_reward",opponent_avg_reward,episode)
                self.logger.add_scalar("win_rate",win_rate,episode)
                print("episode: ", episode, "my_avg_reward:", my_avg_reward, "opponent_avg_reward:", opponent_avg_reward, "win_rate:", win_rate)
            if episode % g_args("update_target_frequency") == 0: #更新target网络
                self.runner.agent.update_targets()
            episode += 1
        self.save()

    def save(self):
        self.runner.agent.save_models(f"models/qmix/model_{self.name}_{self.train_idx}") #路径可修改
           
    def test(self):
        self.runner.setup(self.learner, self.opponent)
        self.runner.set_render()
        self.runner.test(1)

    def get_game_results(self):
        return self.runner.get_game_results()

    def init_logger(self):
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        #画图
        writer_path = f"logs/qmix/{current_time}_{self.name}_{self.train_idx}"
        if not os.path.exists(writer_path): 
            os.makedirs(writer_path)
        print(writer_path)
        self.logger = SummaryWriter(log_dir=writer_path)

    def load_from(self, model_name):
        self.learner.load_models(f"models/qmix/{model_name}")

    def load_learner(self, model):
        self.learner.load_dict(model)

