import collections
import numpy as np

from copy import deepcopy
from learner import create_learner
from learner.mat import MATLearner
from learner.qmix import QMixLearner
from learner.mappo import MAPPOLearner
from config import global_args as g_args


class League:
    def __init__(self, base_model, main_agents, main_exploiters, league_exploiters):
        self.base_model = deepcopy(base_model)
        self.payoff = Payoff()
        self.players = []
        # 初始化联邦中的三个种群  MP ME LE
        l = [MainPlayer(self.base_model, self.payoff, idx+1) for idx in range(main_agents)]
        [self.payoff.add_player(a.checkpoint()) for a in l] # 加入主智能体的初始化状态作为历史记录
        self.players.extend(l)
        l = [MainExploiter(self.base_model, self.payoff, idx+1) for idx in range(main_exploiters)]
        self.players.extend(l)
        l = [LeagueExploiter(self.base_model, self.payoff, idx+1) for idx in range(league_exploiters)]
        self.players.extend(l)
        # 将联邦中的个体都加入对战表中
        for player in self.players:
            self.payoff.add_player(player)

    def update_results(self, home, away, results):
        return self.payoff.update(home, away, results)

    def get_all_players(self):
        return self.players

    def add_player(self, player):
        self.payoff.add_player(player)


class Player:
    def __init__(self, base_model, payoff):
        self.name = "Player"
        self.base_model = deepcopy(base_model)
        self.learner = create_learner()
        self.learner.load_dict(self.base_model)
        self.payoff = payoff
        self.total_step = 0

    def get_match(self):
        pass

    def ready_to_checkpoint(self):
        return False

    def reset_model(self):
        self.learner.load_dict(self.base_model)

    def get_actions(self, d):
        # TODO QMIX 支持 
        #if isinstance(self.learner, QMixLearner):
        #    return self.learner.choose_actions(d["obs"], d["last_action"], d["valid_action"], d["global_steps"], d["training"])
        if isinstance(self.learner, MAPPOLearner):
            return self.learner.get_actions_from_raw_obs(d["obs"])
        elif isinstance(self.learner, MATLearner):
            return self.learner.get_actions_from_raw_obs(d["obs"])

    def create_history(self):
        return Historical(self)

    def checkpoint(self):
        raise NotImplementedError

    def get_model(self):
        return self.learner.save_dict()

    def get_name(self):
        return self.name

    def add_step(self, steps):
        self.total_step += steps


class MainPlayer(Player):
    def __init__(self, base_model, payoff, idx):
        super().__init__(base_model, payoff)
        self.name = f"MP_{idx}"
        self.generation = 1 # 第几代玩家
        self.checkpoint_step = 0
        print(f"创建{self.name}")

    def get_match(self):
        p = np.random.random()
        rand_func = lambda l, r: np.random.choice(l, p=pfsp(r, weighting="squared"))
        # 50% 的几率从联盟中所有历史对手中挑战选手
        if p < 0.5:
            historical = self.payoff.get_historical()
            win_rates = self.payoff[self, historical]
            return rand_func(historical, win_rates)

        main_agents = self.payoff.get_main_players()
        opponent = np.random.choice(main_agents)
        his_opponent = self.payoff.get_historical_opponent(opponent)
        # 35% 的概率与MP对战
        if p < 0.85 and self.payoff[self, opponent] > 0.3:
            return opponent
        # 15% 的几率打ME的历史中胜率低的或者MP的历史
        else:
            historical = self.payoff.get_historical_main_exploiter()
            win_rates = self.payoff[self, historical]
            if len(win_rates) and win_rates.min() < 0.3:
                return rand_func(historical, win_rates)
            win_rates, historical = remove_monotonic_suffix(win_rates, his_opponent) 
            if len(win_rates) and win_rates.min() < 0.7:
                return rand_func(historical, win_rates)
        #兜底选opponent的历史
        win_rates = self.payoff[self, his_opponent]
        return np.random.choice(his_opponent, p=pfsp(win_rates, weighting="variance"))

    def ready_to_checkpoint(self):
        # 步数大于X或胜率大于0.7时可以存档
        steps_passed = self.total_step - self.checkpoint_step
        if steps_passed < g_args("step_passed"):
            return False
        historical = self.payoff.get_historical()
        win_rates = self.payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed >g_args("step_passed2")

    def checkpoint(self):
        self.checkpoint_step = self.total_step
        obj = self.create_history()
        self.generation += 1
        print(f"存档！{self.name} 进化至第{self.generation} 代")
        return obj


class LeagueExploiter(Player):
    def __init__(self, base_model, payoff, idx):
        super().__init__(base_model, payoff)
        self.name = f"LE_{idx}"
        self.generation = 1
        self.checkpoint_step = 0
        print(f"创建{self.name}")

    def get_match(self):
        # 按照pfsp与全联盟中的智能体对战
        historical = self.payoff.get_historical()
        win_rates = self.payoff[self, historical]
        return np.random.choice(historical, p=pfsp(win_rates, weighting="linear_capped"))

    def checkpoint(self):
        # 和原文伪代码不同，原文是存档时25%的概率重置参数，即是存到历史的玩家有可能是重置模型
        # 历史模型为什么要存重置的呢？故这里改为先生成历史模型，再25%的概率重置现在的模型
        self.checkpoint_step = self.total_step
        historical = self.create_history()
        self.generation += 1
        print(f"存档！{self.name} 进化至第{self.generation} 代")
        if np.random.random() < 0.25:
            self.reset_model()
        return historical

    def ready_to_checkpoint(self):
        # 步数大于X或胜率大于0.7时可以存档
        steps_passed = self.total_step - self.checkpoint_step
        if steps_passed < g_args("step_passed"):
            return False
        historical = self.payoff.get_historical()
        win_rates = self.payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > g_args("step_passed2")


class MainExploiter(Player):
    """
    能打败MainPlayer中的所有智能体
    """
    def __init__(self, base_model, payoff, idx):
        super().__init__(base_model, payoff)
        self.name = f"ME_{idx}"
        self.generation = 1
        self.checkpoint_step = 0
        print(f"创建{self.name}")

    def get_match(self):
        # 如果能以大于0.1的胜率打败当前MP中的对手则与之训练
        main_agents = self.payoff.get_main_players()
        opponent = np.random.choice(main_agents)
        if self.payoff[self, opponent] > 0.1:
            return opponent
        # 否则就从当前训练中的MP的祖先中，按照pfsp挑选手
        his_opponent = self.payoff.get_historical_opponent(opponent)
        win_rates = self.payoff[self, his_opponent]
        return np.random.choice(his_opponent, p=pfsp(win_rates, weighting="variance"))

    def ready_to_checkpoint(self):
        # 当步数超上限了或以0.7的胜率打败全部3个正在学习的主智能体
        steps_passed = self.total_step - self.checkpoint_step
        if steps_passed < g_args("step_passed"):
            return False
        main_agents = self.payoff.get_main_players()
        win_rates = self.payoff[self, main_agents]
        return win_rates.min() > 0.7 or steps_passed > g_args("step_passed2")

    def checkpoint(self):
        self.checkpoint_step = self.total_step
        historical = self.create_history()
        self.generation += 1
        print(f"存档！{self.name} 进化至第{self.generation} 代")
        self.reset_model()
        return historical

class Historical(Player):
    def __init__(self, player):
        self.parent = player
        super().__init__(player.get_model(), player.payoff)
        self.name = f"historical of {self.parent.get_name()} gen {self.parent.generation}"

    def create_history(self):
        raise ValueError("history should not create history")

class Payoff:
    def __init__(self) -> None:
        self.players = []
        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)
        self.decay = 0.99

    def __getitem__(self, match):
        home, away = match
        home = [home] if isinstance(home, Player) else home
        away = [away] if isinstance(away, Player) else away
        win_rates = np.array([ [self.win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)
        return win_rates

    def win_rate(self, home, away):
        if self.games[home, away] == 0:
            return 0.5
        return (self.wins[home, away] + self.draws[home, away] * 0.5) / self.games[home, away]

    def update(self, home, away, results):
        for stats in (self.games, self.wins, self.draws, self.losses):
            stats[home, away] *= self.decay
            stats[away, home] *= self.decay

        for result in results:
            self.games[home, away] += 1
            self.games[away, home] += 1
            if result == "win":
                self.wins[home, away] += 1
                self.losses[away, home] += 1
            elif result == "draw":
                self.draws[home, away] += 1
                self.draws[away, home] += 1
            else:
                self.wins[away, home] += 1
                self.losses[home, away] += 1

    def add_player(self, player):
        self.players.append(player)

    def get_historical(self):
        return [p for p in self.players if isinstance(p, Historical)]

    def get_main_players(self):
        return [p for p in self.players if isinstance(p, MainPlayer)]

    def get_main_exploiter(self):
        return [p for p in self.players if isinstance(p, MainExploiter)]

    def get_historical_main_exploiter(self):
        historical = self.get_historical()
        return [p for p in historical if isinstance(p.parent, MainExploiter)]

    def get_historical_main_player(self):
        historical = self.get_historical()
        return [p for p in historical if isinstance(p.parent, MainPlayer)]

    def get_historical_opponent(self, opponent):
        historical = self.get_historical()
        return [p for p in historical if p.parent == opponent]

    def get_historical_win_rate(self, home, opponent):
        historical = self.get_historical_opponent(opponent)
        d = {}
        for p in historical:
            d[f"[{home.get_name()}] vs [{p.get_name()}]"] = self[home, p]
        return d

    def get_win_rate(self, home, opponent):
        d = {}
        d[f"[{home.get_name()}] vs [{opponent.get_name()}]"] = self[home, opponent]
        return d



def pfsp(win_rates, weighting="linear"):
    weightings = {
        "variance": lambda x: x * (1 - x),  #胜率越接近0.5，选择概率越高
        "linear": lambda x: 1 - x,          #胜率越高，选择概率越低
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x)**2,    #胜率越高，选择概率更低
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm

def remove_monotonic_suffix(win_rates, players):
    if not win_rates:
        return win_rates, players
    for i in range(len(win_rates) - 1, 0, -1):
        if win_rates[i - 1] < win_rates[i]:
            return win_rates[:i + 1], players[:i + 1]
    return np.array([]), []
