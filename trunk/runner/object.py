import pprint
import torch
import numpy as np
from buffer import ReplayBuffer, SharedReplayBuffer, ExpertBuffer
import random
from learner.mat import MATLearner
from utils.util import prettyprint
from env.chooseenv import make, make_parallel_env
from agents.random.submission import my_controller as rand_control
from agents.football_5v5_mappo.submission import my_controller as ppo_control_5
from agents.football_11v11_mappo.submission import my_controller as ppo_control_11
from agents.football_5v5_qmix import submission as sub
from config import global_args as g_args
from learner import create_learner, get_learner
from alphastar.selfplay import Player
from rewarder.rewarder_wekick import reward_warpper
def _t2n(x):
    return x.detach().cpu().numpy()

class EpisodeRunner:

    def __init__(self) -> None:
        self.render = False
        self.buffer = ReplayBuffer() 
        self.agent = None
        self.env = make(g_args("env_name")) 
        self.opponent = ""
        self.gloabl_steps = 0 #整个训练流程中的总步数,用于epsilon decay 
        self.batch_size = g_args("batch_size")
        self.game_results = []

    def setup(self, learner, opponent):
        self.agent = learner
        self.opponent = opponent

    def set_render(self, render=True):
        self.render=render

    def run(self):
        self.agent.init_hidden_states(1)#init hidden state, batch_size=1
        all_observes_pre = self.env.all_observes
        state, obs = self.reset()
        valid_action = get_valid_action(obs)
        last_action = [[0] for _ in range(g_args("n_agents"))] #   (n_agent,action_dim)
        reward = None
        terminated = False
        episode_data = []
        while not terminated:
            my_ai_action =  self.agent.choose_actions( obs, last_action, valid_action, self.gloabl_steps, True) #[num_agent, action_dim] (4,1)
            if isinstance(self.opponent, Player):
                info = {"obs": obs, "last_action": last_action, "valid_action": valid_action, "global_steps": self.gloabl_steps, "training": False}
                self.opponent.get_actions(info)
            else:
                joint_action = self.get_joint_actions(my_ai_action)
            all_observes,reward,terminated,_,_ = self.env.step(joint_action)
            next_state = get_state(all_observes)
            next_obs = get_obs(all_observes[0:g_args("n_agents")])#[n_agent, obs_dim]  
            next_valid_action = get_valid_action(next_obs)
            #只计算了我方的reward shaping  
            joint_action_decode = decode(joint_action)
            shaped_reward = reward_adapter(reward, all_observes_pre, all_observes, joint_action_decode)
            reward = sum(shaped_reward)                                       
            data = {}
            data["state"] = state   # 双方8个球员看到的信息 
            data["obs"] = obs  # 我方4个球员看到的消息
            data['valid_action'] = valid_action
            data["last_action"] = last_action
            data["action"] =  my_ai_action
            data["next_state"] = next_state
            data["next_obs"] = next_obs
            data["next_valid_action"] = next_valid_action
            data["reward"] = [reward]
            data["terminated"] = [terminated]
            episode_data.append(data)
            #update   
            self.gloabl_steps += 1
            state = next_state
            obs = next_obs
            last_action =  my_ai_action
            valid_action = next_valid_action
            all_observes_pre = all_observes
        self.game_results.append(self.env.check_win())
        self.buffer.add(episode_data)

    def can_sample(self):
        return self.buffer.can_sample(self.batch_size)

    def sample(self):
        return self.buffer.sample(self.batch_size)
 
    def get_opponent_actions(self):
        multi_part_agent_ids, multi_part_actions_space = get_players_and_action_space_list(self.env)
        joint_action = []
        for agent_ids, action_space in zip(multi_part_agent_ids, multi_part_actions_space):
            for idx, agent_id in enumerate(agent_ids):
                a_obs = self.env.all_observes[agent_id]
                if self.opponent == "football_5v5_mappo":
                    a_act = ppo_control_5(a_obs, action_space[idx])#random 
                else:
                    a_act = rand_control(a_obs, action_space[idx])#random 
                joint_action.append(a_act)
        return joint_action[ g_args("n_agents"): g_args("n_agents")*2] # 只返回对方的动作

    def get_joint_actions(self, my_ai_action ):
        shaped_my_ai_action = get_shaped_actions(my_ai_action)
        # print("shaped_my_ai_action:", shaped_my_ai_action)
        # 获取对方动作
        opponent_action = self.get_opponent_actions() #         
        '''
        opponent_action 格式, 和agent的数量有关，4个agent即有4组动作
        [
         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], 
         [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], 
         [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
         [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]
        '''
        joint_action = shaped_my_ai_action + opponent_action
        return joint_action

    def test(self,test_episodes):
        def test_one_episode(): #进行一局游戏，返回比分,reward是raw reward
            self.agent.init_hidden_states(1)
            state, obs = self.reset()
            last_action  =  [[0]for _ in range(4)] #   (n_agent,action_dim)
            valid_action = get_valid_action(obs) 
            my_total_reward = 0
            opponent_total_reward = 0
            terminated = False
            while not terminated:
                my_ai_action =  self.agent.choose_actions( obs, last_action, valid_action, 0, False) # greedy strategy
                joint_action = self.get_joint_actions(my_ai_action)
                all_observes, reward, terminated, _, _ = self.env.step(joint_action)
                if self.render and not terminated:
                    self.env.env_core.render()
                next_state = get_state(all_observes)
                next_obs = get_obs(all_observes[0:g_args("n_agents")] )
                next_valid_action = get_valid_action(next_obs)
                my_total_reward += sum(reward[0:g_args("n_agents")])
                opponent_total_reward += sum(reward[g_args("n_agents"): g_args("n_agents")*2])
                state = next_state
                obs = next_obs
                last_action =  my_ai_action
                valid_action = next_valid_action
            winner = self.env.check_win()#'0'表示我方赢 '1'表示对方赢 '-1'表示平局
            if winner == '0':
                isWin = True
            else:
                isWin = False

            print("my_total_reward:", my_total_reward, "opponent_total_reward:", opponent_total_reward, "Win:", isWin)
            return my_total_reward, opponent_total_reward,isWin

        #进行多局游戏，返回平均得分和胜率
        my_total_reward = 0
        opponent_total_reward = 0
        win_episode = 0
        for _  in range(test_episodes):
            my_reward, opponent_reward, win = test_one_episode()
            my_total_reward += my_reward
            opponent_total_reward += opponent_reward
            if win:
                win_episode += 1             
        return my_total_reward/test_episodes, opponent_total_reward/test_episodes, win_episode/test_episodes  #返回平均得分和胜率

    def reset(self):
        # self.env.reset()
        self.env = make( g_args("env_name"))  #TODO env如何reset
        return get_state(self.env.all_observes) ,get_obs(self.env.all_observes[: g_args("n_agents")])  

    def close_env(self):
        pass

    def timestep(self):
        return self.gloabl_steps 

    def get_game_results(self):
        ret = ["win" if r == "0" else "loss" for r in self.game_results]
        return ret


class MultiThreadRunner:

    def __init__(self) -> None:
        self.opponent = ""
        self.learner = None
        self.game_results = []
        self.thread_num = g_args("thread_num") 
        self.env = make_parallel_env()
        self.eval_env = make_parallel_env()
        self.episode_length = g_args("episode_length")
        self.eval_episode_length = g_args("eval_episode_length")
        self.left_agent_num = g_args("left_agent_num")
        self.right_agent_num = g_args("right_agent_num")
        self.use_centralized_V = g_args("use_centralized_V")
        self.eval_episode_num = g_args("eval_episode_num")
        self.recurrent_N = g_args("recurrent_N")
        self.hidden_size = g_args("hidden_size")
        self.use_reward_wrapper = g_args("use_reward_warpper")
        self.total_agent = self.left_agent_num + self.right_agent_num
        self.share_observation_space = self.env.share_observation_space[0] if self.use_centralized_V else self.env.observation_space
        self.observation_space = self.env.observation_space
        self.private_observation_space = self.env.private_observation_space
        self.action_space = self.env.action_space
        self.buffer = SharedReplayBuffer(self.observation_space, self.share_observation_space, self.private_observation_space, self.action_space)
        # warmup
        obs,private_obs = self.env.reset()
        obs = obs[:,0:self.left_agent_num,:]
        private_obs = private_obs[:,0:self.left_agent_num,:]
        share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.private_obs[0] = private_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def setup(self, learner, opponent):
        self.learner = learner
        self.opponent = opponent

    def run(self):
        pre_raw_obs = self.env.get_current_states()
        for step in range(self.episode_length):
            data = {}
            data['pre_raw_obs'] = pre_raw_obs
            collects, opp_actions = self.collect(step)
            obs, private_obs, rewards, dones, win = self.step(self.env, collects["actions"], opp_actions)
            raw_obs = self.env.get_current_states()
            data.update(collects)
            data['raw_obs'] = raw_obs
            data['opp_actions'] = opp_actions
            data["obs"] = obs
            data["private_obs"] = private_obs
            data['rewards'] = rewards
            data["dones"] = dones
            #reward shaping
            if(self.use_reward_wrapper): data['rewards'] = reward_warpper(data)
            pre_raw_obs = raw_obs
            # print(step)
            # print(collects["actions"])
            # print(rewards)
            # print(dones)
            masks = np.ones((self.thread_num, self.left_agent_num, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
            self.insert(data, masks)
            
        return self.buffer

    @torch.no_grad()
    def eval(self):
        print("---------begin eval-------------")
        self.learner.prep_rollout()
        episodes_rewards = []
        win_history = []
        obs, _ = self.eval_env.reset()
        obs = obs[:, 0:self.left_agent_num, :]

        masks = np.ones((self.thread_num, self.left_agent_num, 1), dtype=np.float32)
        episodes = self.eval_episode_num // self.thread_num
        for episode in range(episodes):
            episode_rewards = []
            for step in range(self.eval_episode_length):
                available_actions = np.array(obs[...,: self.eval_env.action_space.n])
                obs = np.concatenate(obs)
                actions, _ = self.learner.get_actions(obs, available_actions)
                #取对方动作
                opp_actions = []
                current_states = self.eval_env.get_current_states()
                for threads in range(self.thread_num):
                    thr_opp_actions = []
                    for obs_id in range(self.left_agent_num, self.total_agent):
                        state = current_states[threads][obs_id]
                        opp_action = self.get_opp_actions(state)
                        thr_opp_actions.append(opp_action)
                    decode_action = self.eval_env.decode(thr_opp_actions)
                    opp_actions.append(decode_action)

                obs, _, rewards, dones, won_mask = self.step(self.eval_env, actions, opp_actions)
                episode_rewards.append(rewards)
                masks = np.ones((self.thread_num, self.left_agent_num, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            avg_episode_rewards = np.sum(np.array(episode_rewards), axis=0)
            avg_episode_rewards = np.mean(avg_episode_rewards)
            episodes_rewards.append(avg_episode_rewards)
            win_history.append(won_mask)

        env_infos = {}
        win_history = np.array(win_history,dtype=float).reshape(-1)
        self.game_results.extend([r for r in win_history])
        env_infos['eval_average_episode_rewards'] = np.array(episodes_rewards)
        env_infos['winnning rate'] = np.maximum(win_history, 0)
        env_infos['winnning rate(with tie game)'] = np.maximum(win_history, -win_history)
        return env_infos

    def get_opp_actions(self, state):
        action_space = self.env.joint_action_space[self.left_agent_num:]
        if isinstance(self.opponent, Player):
            action_space = self.env.joint_action_space[self.left_agent_num:]
            d = {}
            d["obs"] = state
            opp_action = self.opponent.get_actions(d)
        elif self.opponent == "buildin_ai":
            opp_action = [0]*20
            opp_action[19] = 1
            opp_action = [opp_action]
        elif self.opponent == "football_5v5_mappo":
            opp_action = ppo_control_5(state, action_space) 
        elif self.opponent == "football_11v11_mappo":
            opp_action = ppo_control_11(state, action_space)
        else:
            opp_action = rand_control(state, action_space)
        return opp_action

    @torch.no_grad()
    def collect(self, step):
        self.learner.prep_rollout()
        #get agent actions
        obs = self.buffer.get_obs(step)
        cent_obs = self.buffer.get_share_obs(step)
        available_actions = self.buffer.get_available_actions(step)
        if isinstance(self.learner, MATLearner):
            values = self.learner.get_values(obs)
        else:
            values = self.learner.get_values(cent_obs)
        actions, action_probs = self.learner.get_actions(obs, available_actions)
   
        #get opponent actions
        opp_actions = []
        current_states = self.env.get_current_states()
        for threads in range(self.thread_num):
            thr_opp_actions = []
            for obs_id in range(self.left_agent_num, self.total_agent):
                state = current_states[threads][obs_id]
                opp_action = self.get_opp_actions(state)
                thr_opp_actions.append(opp_action)
            decode_action = self.env.decode(thr_opp_actions)
            opp_actions.append(decode_action)

        data = {}
        data["values"] = values
        data["actions"] = actions
        data["action_log_probs"] = action_probs
        data["available_actions"] = available_actions
        return data, opp_actions

    def step(self, env, actions, opp_actions):
        all_actions = np.squeeze(actions, axis=-1).tolist()
        for thread in range(self.thread_num):
            all_actions[thread] = all_actions[thread] + opp_actions[thread]
        obs, private_obs, rewards, dones, won_mask, _ = env.step(all_actions)
        obs = obs[:, 0:self.left_agent_num, :]
        private_obs = private_obs[:, 0:self.left_agent_num, :]
        rewards = rewards[:, 0:self.left_agent_num]
        dones = dones[:, 0:self.left_agent_num]
        return obs, private_obs, rewards, dones, won_mask

    def insert(self, data, masks):
        self.buffer.insert(data, masks)

    def clear_game_results(self):
        self.game_results = []

    def get_game_results(self):
        print("游戏结果", self.game_results)
        ret = []
        for r in self.game_results:
            if r == 1:
                ret.append("win")
            elif r == -1:
                ret.append("draw")
            else:
                ret.append("loss")
        return ret


def get_state(raw_all_obs): 
    encoder = sub.FeatureEncoder()
    state =  []
    for i in raw_all_obs:
        state .extend(  sub.concate_observation_from_raw(encoder.encode(i)))
    return state #[  state_dim,]

def get_obs(partial_raw_obs):
    encoder = sub.FeatureEncoder()
    obs = [sub.concate_observation_from_raw(encoder.encode(d)) for d in partial_raw_obs]
    return obs #[num_agent,obs_dim]

def get_valid_action(obs):
    return np.array([o[:19] for o in obs])

def get_shaped_actions(action):
    shaped_action = []
    for i in action:
        act = [0]*g_args("n_actions")#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        act[i[0]] = 1
        shaped_action.append( [act])
    return shaped_action
 
def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space

def decode(joint_action):
    if isinstance(joint_action, np.ndarray):
        joint_action = joint_action.tolist()
    joint_action_decode = []
    for action in joint_action:
        joint_action_decode.append(action[0].index(1))
    return joint_action_decode

#obs视角当自己是左方,返回我方的reward_list
def reward_adapter(reward, raw_obs_pre, raw_obs_now, joint_action):
    # 只计算我方的数据
    agent_num = g_args("n_agent")
    reward = reward[0:agent_num]
    raw_obs_pre = raw_obs_pre[0:agent_num]
    raw_obs_now = raw_obs_now[0:agent_num]
    action = joint_action[0:agent_num]
    opponent_action = joint_action[agent_num:-1]
    for i in range(len(reward)):
        reward[i] = get_reward(reward[i], raw_obs_pre[i], raw_obs_now[i] ,action[i], opponent_action)
    return reward

def get_reward(reward, obs_pre, obs_now, action, opponent_action): #action是我方一个agent的动作，opponent_action是对方所有agent的动作
    active = obs_pre['active']#我方球员编号
    ball_owned_player_pre = obs_pre['ball_owned_player']#持球队员编号
    ball_owned_player_now = obs_now['ball_owned_player']
    ball_owned_team_pre = obs_pre['ball_owned_team']#持球队伍,0本队，1对手
    ball_owned_team_now = obs_now['ball_owned_team']
    #判断是否控球
    if ball_owned_team_pre==0 and ball_owned_player_pre==active:
        control_ball_pre = True
    else:
        control_ball_pre = False
    if ball_owned_team_now==0 and ball_owned_player_now==active:
        control_ball_now = True
    else:
        control_ball_now = False
    #控球奖励
    if not control_ball_pre and control_ball_now: #获得控球
        reward += 0.2
    if control_ball_pre and not control_ball_now: #失去控球
        reward -= 0.2
    #铲球奖励
    if action == 16 and ball_owned_team_pre == 1 and ball_owned_team_now != 1: #我方铲球成功
        reward += 0.2
    if 16 in opponent_action and control_ball_pre and not control_ball_now: #对方铲球成功
        reward -= 0.2 
    #带球奖励
    if ball_owned_team_now == 0:
        reward += 0.001 
    if ball_owned_team_now == 1:#对方带球
        reward -= 0.001 
    return reward

