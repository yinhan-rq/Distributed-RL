import threading
from time import time
from config import global_args as g_args
import trainer
import copy

from utils.util import prettyprint


class ActorLoop:
    """
    用来在自博弈中组织发生对战
    并把对战信息发给learner，使之学习
    """

    def __init__(self, player, league):
        self.player = player
        self.league = league
        self.trainer = trainer.create_trainer("mappo")
        self.training_time = g_args("training_time")
        self.thread = threading.Thread(target=self.run)
        self.trainer.set_name(self.player.get_name())

    def start(self):
        self.thread.start()

    def run(self):
        start_time = time()
        while time() - start_time < self.training_time:
            opp = self.player.get_match()
            self.opponent = copy.deepcopy(opp)
            print(f"{self.player.get_name()} 开启第{self.trainer.train_idx + 1}次对战vs {self.opponent.get_name()}")
            prettyprint(self.player.payoff.get_win_rate(self.player, opp), "战前胜率")
            prettyprint(self.player.payoff.get_historical_win_rate(self.player, opp), " 历史胜率")
            # 开启一轮训练
            self.trainer.set_opponent(self.opponent)
            self.trainer.train()
            steps = self.trainer.get_steps()
            self.player.add_step(steps)
            # 每一个对手的多次对局中，记录更新所有对局结果
            results = self.trainer.get_game_results()
            print(f"{self.player.get_name()} 结束第{self.trainer.train_idx}次对战vs {self.opponent.get_name()}对战结果 {results}")
            self.league.update_results(self.player, opp, results)
            if self.player.ready_to_checkpoint():
                self.league.add_player(self.player.checkpoint())
