# -*- coding:utf-8  -*-
# Time  : 2020/12/28 16:33
# Author: Yahui Cui
import copy
from pprint import pprint

from env.simulators.game import Game
from env.obs_interfaces.observation import *
import gfootball.env as football_env
import numpy as np
import json
from utils.discrete import Discrete
from utils.box import Box

import gym
from gym import spaces
from utils.encoder import *

class Football(Game, DictObservation):
    def __init__(self, conf):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.env_core = football_env.create_environment(
            env_name=conf["game_name"], stacked=False,
            representation='raw',
            logdir='/tmp/rllib_test',
            write_goal_dumps=False, write_full_episode_dumps=True, render=False,
            dump_frequency=0,
            rewards='scoring,checkpoints',
            number_of_left_players_agent_controls=self.agent_nums[0],
            number_of_right_players_agent_controls=self.agent_nums[1])

        self.load_action_space(conf)
        obs_list = self.env_core.reset()
        self.won = {}
        self.joint_action_space = self.set_action_space()
        self.current_state = self.get_sorted_next_state(obs_list)
        self.all_observes = self.current_state
        self.n_return_temp = [0.0] * self.n_player
        self.n_return = [0.0] * self.n_player

        self.action_dim = self.get_action_dim()
        self.init_info = None

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if self.is_act_continuous:
                if ("high" not in input_action) or ("low" not in input_action) or ("shape" not in input_action):
                    raise Exception("act_box in continuous case must have fields low, high, shape")
                shape = tuple(input_action["shape"])
                self.env_core.action_space = Box(input_action["low"], input_action["high"], shape, np.float32)
            else:
                if "discrete_n" not in input_action:
                    raise Exception("act_box in discrete case must have field discrete_n")
                discrete_n = int(input_action["discrete_n"])
                self.env_core.action_space = Discrete(discrete_n)

    def step(self, joint_action):
        action = self.decode(joint_action)
        info_before = self.step_before_info()
        next_state, reward, self.done, info_after = self.get_next_state(action)
        sorted_next_state = self.get_sorted_next_state(next_state)
        self.current_state = sorted_next_state
        self.all_observes = self.current_state
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        reward = self.get_reward(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        return self.all_observes, reward, done, info_before, info_after

    def decode(self, joint_action):
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        joint_action_decode = []
        for action in joint_action:
            joint_action_decode.append(action[0].index(1))
        return joint_action_decode

    def get_next_state(self, action):
        observation, reward, done, info = self.env_core.step(action)
        return observation, reward, done, info

    def get_reward(self, reward):
        r = [0] * self.n_player
        for i in range(self.n_player):
            r[i] = reward[i]
            self.n_return_temp[i] += r[i]

        # left n_return
        self.n_return[0] = self.n_return_temp[0]
        # right n_return
        self.n_return[self.agent_nums[0]] = self.n_return_temp[self.agent_nums[0]]

        return r

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True

        if self.done:
            self.env_core.close()

        return self.done

    def set_action_space(self):
        action_space = [[self.env_core.action_space] for _ in range(self.n_player)]
        return action_space

    def check_win(self):
        left_sum = self.n_return[0]
        right_sum = self.n_return[self.agent_nums[0]]
        if left_sum > right_sum:
            return '0'
        elif left_sum < right_sum:
            return '1'
        else:
            return '-1'

    def reset(self):
        obs_list = self.get_sorted_next_state(self.env_core.reset())
        self.step_cnt = 0
        self.done = False
        self.current_state = obs_list
        self.all_observes = self.current_state
        return self.all_observes

    def get_action_dim(self):
        action_dim = 1
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space)):
            action_dim *= self.joint_action_space[i][0].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_render_data(self, current_state):
        return []

    def get_dict_observation(self, current_state, player_id, info_before):
        ob = current_state[player_id]
        ob['controlled_idx'] = player_id
        return ob

    def get_sorted_next_state(self, next_state):
        left_team = next_state[:self.agent_nums[0]]
        right_team = next_state[self.agent_nums[0]:]
        left_team = sorted(left_team, key=lambda keys: keys['active'])
        right_team = sorted(right_team, key=lambda keys: keys['active'])

        new_state = []
        index = 0
        for item in left_team:
            each = copy.deepcopy(item)
            each["controlled_player_index"] = index
            new_state.append(each)
            index += 1

        for item in right_team:
            each = copy.deepcopy(item)
            each["controlled_player_index"] = index
            new_state.append(each)
            index += 1

        return new_state

class Football2(Game, DictObservation):
    def __init__(self, conf, dump_dir=None):
        super().__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], conf['agent_nums'], conf['obs_type'])

        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])

        self.env_core = football_env.create_environment(
            env_name=conf["game_name"], stacked=False,
            representation='raw',
            logdir=dump_dir if dump_dir else './IL_test',
            write_goal_dumps=False, write_full_episode_dumps=True, render=False,
            dump_frequency=1 if dump_dir else 100,
            rewards='scoring,checkpoints',
            number_of_left_players_agent_controls=self.agent_nums[0],
            number_of_right_players_agent_controls=self.agent_nums[1],
            other_config_options= {
                "action_set" : "v2"
            })
        self.load_action_space(conf)
        obs_list = self.env_core.reset()
        self.obs_list = obs_list
        self.won = 0
        self.joint_action_space = self.set_action_space()
        self.action_space = self.get_single_action_space(0)
        self.current_state = self.get_sorted_next_state(obs_list)   #raw joint obeservation
        # transfer observation
        self.encoder = FeatureEncoder()
        self.all_observes = []
        self.private_all_observes = []
        for observe in self.current_state:
            encode_obs = concate_observation_from_raw(self.encoder.encode(observe))
            private_encode_obs = concate_observation_from_raw(self.encoder.single_agent_encode(observe))
            self.all_observes.append(encode_obs)
            self.private_all_observes.append(private_encode_obs)
        self.all_observes = np.array(self.all_observes)    #transfered observation  list[array(133,)], len(list)=n_players
        self.private_all_observes = np.array(self.private_all_observes)
        self.all_observes_dim = self.all_observes[0].size
        self.private_observes_dim = self.private_all_observes[0].size
        # contrust observation space
        low = np.array([np.inf]*self.all_observes_dim)
        high = np.array([-np.inf]*self.all_observes_dim)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        # contrust private observation space
        low = np.array([np.inf]*self.private_observes_dim)
        high = np.array([-np.inf]*self.private_observes_dim)
        self.private_observation_space = gym.spaces.Box(low, high, dtype=np.float64)
        #all observation space
        self.joint_observation_space = [self.observation_space for _ in range(self.n_player)]
        self.share_observation_space = self.joint_observation_space.copy()

        self.n_return_temp = [0.0] * self.n_player
        self.n_return = [0.0] * self.n_player
        self.action_dim = self.get_action_dim()
        self.init_info = None

        self.role_mask = False
    # random seed
    def seed(self, seed=None):
        if seed is None:
            self.env_core.seed(1)
        else:
            self.env_core.seed(seed)

    def load_action_space(self, conf):
        if "act_box" in conf:
            input_action = json.loads(conf["act_box"]) if isinstance(conf["act_box"], str) else conf["act_box"]
            # print(input_action)
            if self.is_act_continuous:
                if ("high" not in input_action) or ("low" not in input_action) or ("shape" not in input_action):
                    raise Exception("act_box in continuous case must have fields low, high, shape")
                shape = tuple(input_action["shape"])
                self.env_core.action_space = Box(input_action["low"], input_action["high"], shape, np.float32)
            else:
                if "discrete_n" not in input_action:
                    raise Exception("act_box in discrete case must have field discrete_n")
                discrete_n = int(input_action["discrete_n"])
                self.env_core.action_space = spaces.Discrete(discrete_n)

    def step(self, joint_action):
        self.won = self.check_win()
        #记录上一观测的得分和比赛模式，用于校正角色
        last_score = self.current_state[0]['score']
        last_game_mode = self.current_state[0]['game_mode']
        next_state, reward, self.done, info_after = self.get_next_state(joint_action)
        self.current_state = self.get_sorted_next_state(next_state)
        #获取这一步的得分和比赛模式
        score = self.current_state[0]['score']
        game_mode = self.current_state[0]['game_mode']
        #和上一步比较
        self.score_change_mask = False if last_score == score else True
        self.game_mode_change_mask = False if last_game_mode == game_mode else True
        if(self.score_change_mask): self.role_mask = False   #用stochastic的role
        if(self.game_mode_change_mask): self.role_mask = True  #用kaggle的role
        self.all_observes = []
        self.private_all_observes = []        
        for observe in self.current_state:
            encode_obs = concate_observation_from_raw(self.encoder.encode(observe, role = self.role_mask))
            private_encode_obs = concate_observation_from_raw(self.encoder.single_agent_encode(observe, role = self.role_mask))
            self.all_observes.append(encode_obs)
            self.private_all_observes.append(private_encode_obs)
        self.all_observes = np.array(self.all_observes)
        self.private_all_observes = np.array(self.private_all_observes)
        # pprint("step:{}".format(self.step_cnt))
        # pprint("score_change:{}".format(self.score_change_mask))
        # pprint("game_mode_change:{}".format(self.game_mode_change_mask))
        # pprint("oringin_9_role:{}".format(self.current_state[0]['left_team_roles'][9]))
        # pprint("now_9_role{}".format(self.all_observes[9][119:129]))
        # pprint("now_9_private_role{}".format(self.private_all_observes[9][23:33]))
        # pprint("---------------------------------------------------")
        # pprint("now_all_role",)
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        reward = self.get_reward(reward)
        self.step_cnt += 1
        dones = self.is_terminal()
        dones = np.array([dones] * self.n_player)
        return self.all_observes, self.private_all_observes, reward, dones, self.won, info_after

    def decode(self, joint_action):
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        joint_action_decode = []
        for action in joint_action:
            joint_action_decode.append(action[0].index(1))
        return joint_action_decode

    def get_next_state(self, action):
        observation, reward, done, info = self.env_core.step(action)
        return observation, reward, done, info

    def get_reward(self, reward):
        r = [0] * self.n_player
        for i in range(self.n_player):
            r[i] = reward[i]
            # self.n_return_temp[i] += r[i]

        # left n_return
        # self.n_return[0] = self.n_return_temp[0]
        # right n_return
        # self.n_return[self.agent_nums[0]] = self.n_return_temp[self.agent_nums[0]]

        return r

    def step_before_info(self, info=''):
        return info

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            self.done = True           
        # if self.done:
        #     self.env_core.close()
        return self.done

    def set_action_space(self):
        action_space = [self.env_core.action_space for _ in range(self.n_player)]
        return action_space

    def check_win(self):
        left_score = self.current_state[0]["score"][0]
        right_score = self.current_state[0]["score"][1]
        if left_score > right_score:
            return 1
        elif left_score < right_score:
            return 0
        elif left_score == right_score:
            return -1


    def reset(self):
        obs_list = self.get_sorted_next_state(self.env_core.reset())
        self.step_cnt = 0
        self.n_return_temp = [0.0] * self.n_player
        self.n_return = [0.0] * self.n_player
        self.won = 0
        self.done = False
        self.current_state = obs_list
        self.all_observes = []
        self.private_all_observes = []        
        for observe in self.current_state:
            encode_obs = concate_observation_from_raw(self.encoder.encode(observe))
            private_encode_obs = concate_observation_from_raw(self.encoder.single_agent_encode(observe))
            self.all_observes.append(encode_obs)
            self.private_all_observes.append(private_encode_obs)

        self.all_observes = np.array(self.all_observes)
        self.private_all_observes = np.array(self.private_all_observes)
        
        return self.all_observes, self.private_all_observes

    def get_action_dim(self):
        action_dim = 1
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space)):
            action_dim *= self.joint_action_space[i].n

        return action_dim

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_render_data(self, current_state):
        return []

    def get_dict_observation(self, current_state, player_id, info_before):
        ob = current_state[player_id]
        ob['controlled_idx'] = player_id
        return ob

    def get_sorted_next_state(self, next_state):
        left_team = next_state[:self.agent_nums[0]]
        right_team = next_state[self.agent_nums[0]:]
        left_team = sorted(left_team, key=lambda keys: keys['active'])
        right_team = sorted(right_team, key=lambda keys: keys['active'])

        new_state = []
        index = 0
        for item in left_team:
            each = copy.deepcopy(item)
            each["controlled_player_index"] = index
            new_state.append(each)
            index += 1

        for item in right_team:
            each = copy.deepcopy(item)
            each["controlled_player_index"] = index
            new_state.append(each)
            index += 1
        return new_state

    def close(self):
        self.env_core.close()