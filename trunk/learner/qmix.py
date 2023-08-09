import copy
import os
import numpy as np
import torch
from modules.mixer import QMixer
from modules.rnn_network import RNN
from torch.optim import RMSprop
from config import global_args as g_args

class EpsilonGreedy:
    def __init__(self):
        self.epsilon = g_args("epsilon_start")
        self.initial_epsilon = g_args("epsilon_start")
        self.epsilon_end = g_args("epsilon_end")
        self.annealing_steps = g_args("annealing_steps")  
        self.n_actions =  g_args("n_actions")  #可选动作数 足球19
        self.n_agents = g_args("n_agents")

    def act(self, value_action,valid_action):
        if np.random.random() > self.epsilon:
            action = value_action.max(dim=1)[1].cpu().detach().numpy()#返回最大值的索引,即为动作 [6 8 6 2]
        else:
            action =  torch.distributions.Categorical(valid_action).sample().long().cpu().detach().numpy()#随机动作  [3 7 0 4]
        return action

    def epislon_decay(self, step):
        progress = step / self.annealing_steps
        decay = self.initial_epsilon - progress
        if decay <= self.epsilon_end:
            decay = self.epsilon_end
        self.epsilon = decay

class QMixLearner:
    def __init__(self) -> None:
        self.device = g_args("device")
        self.n_agents = g_args("n_agents") #智能体个数
        self.episode_length = g_args("episode_length") #训练时一个episode的长度的最大值
        self.batch_size = g_args("batch_size") #训练时一个batch的大小
        self.gamma = g_args("gamma") #衰减系数
        #MultiAgent
        self.agent = RNN(g_args("obs_dim") +  g_args("action_dim")).to(self.device)
        self.target_agent = copy.deepcopy(self.agent).to(self.device)
        self.target_agent.eval()
        self.hidden_states = None
        self.target_hidden_states = None
        # Mixer
        self.mixer = QMixer().to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_mixer.eval()
        #params
        self.params = list(self.agent.parameters())+list(self.mixer.parameters())
        #optimizer
        self.optimizer = RMSprop(params = self.params, lr = g_args("lr"),alpha= g_args("alpha"), eps=g_args("eps"))
        #epsilon greedy 
        self.epsilon_greedy = EpsilonGreedy()
        
    def choose_actions(self,  obs, last_action,valid_action, global_steps, training=True):
        '''
            根据obs和last action选择动作
            obs np.array [agent_num,obs_dim]
            last_action np.array [agent_num,action_dim]
            valid_action: np.array [agent_num,n_actions] 0 / 1 表示是否可以选择该动作
            return: np.array [agent_num,action_dim]
        '''
        # 从obs中分析得出当前能执行的动作
        obs =  torch.FloatTensor(obs).to(self.device) # [4 133] 
        last_action =  torch.FloatTensor(last_action).to(self.device) #[4 1]
        valid_action = torch.LongTensor(valid_action).to(self.device)#[4 19]
        input = torch.cat( (obs, last_action),-1).to(self.device)
        value_action, self.hidden_states = self.agent(input, self.hidden_states)#[4 19]
        value_action[valid_action==0] = -9999999 #如果不可以选择该动作，则设置为最小值
        if training: # epsilon greedy strategy
            actions = self.epsilon_greedy.act(value_action,valid_action )
            self.epsilon_greedy.epislon_decay(global_steps)
        else: #greedy strategy
            actions = np.argmax(value_action.cpu().data.numpy(), -1)
        actions = actions.reshape(value_action.shape[0], 1) # (num_agents, 1)
        return actions

    def get_data(self, episode_batches):
        """
        [ s1,obs1,ava1, last_act, s2,obs2,ava2, r, t]
        batch 是一个list,长度为batch_size
        每个元素是一个episode,episode是一个list,长度为seq_len ,足球游戏全部一致为3000,每个元素是一个字典
        # 
        #data['state']     (state_dim,) 
        #data['action']   (n_agent,action_dim)
        #data['obs]     (n_agent,obs_dim)
        #data['valid_action']   = (n_agent,n_actions) ] 用0 - 1代表动作是否可选
        """
        def unpack(episode):
            state  = np.array([_['state'] for _ in episode], dtype='float32')
            obs  = np.array([_['obs'] for _ in episode],dtype='float32')
            valid_action  = np.array([_['valid_action'] for _ in episode],dtype='float32')
            last_action = np.array([_['last_action'] for _ in episode],dtype='float32')
            action = np.array([_['action'] for _ in episode] )
            next_state  = np.array([_['next_state'] for _ in episode], dtype='float32')
            next_obs  = np.array([_['next_obs'] for _ in episode],dtype='float32')
            next_valid_action  = np.array([_['next_valid_action'] for _ in episode],dtype='float32')
            reward = np.array([_['reward'] for _ in episode])
            terminated = np.array([_['terminated'] for _ in episode])
            return state, obs, valid_action, last_action, action, next_state, next_obs, next_valid_action, reward, terminated
       
        state_batch, obs_batch, valid_action_batch, last_action_batch,\
        action_batch, \
        next_state_batch, next_obs_batch, next_valid_action_batch, \
        reward_batch, terminated_batch =  [],[],[],[],[],[],[],[],[],[]
        episode_len = len(episode_batches[0])

        for episode in episode_batches:
            s,o,va,la,a,ns,no,nva,r,t = unpack(episode)
            state_batch.append(s)
            obs_batch.append(o)
            valid_action_batch.append(va)
            last_action_batch.append(la)
            action_batch.append(a)
            next_state_batch.append(ns)
            next_obs_batch.append(no)
            next_valid_action_batch.append(nva)
            reward_batch.append(r)
            terminated_batch.append(t)

        state_batch = np.array(state_batch)
        obs_batch = np.array(obs_batch)
        valid_action_batch = np.array(valid_action_batch)
        last_action_batch = np.array(last_action_batch)
        action_batch = np.array(action_batch)
        next_state_batch = np.array(next_state_batch)
        next_obs_batch = np.array(next_obs_batch)
        next_valid_action_batch = np.array(next_valid_action_batch)
        reward_batch = np.array(reward_batch)
        terminated_batch = np.array(terminated_batch)
        
        return state_batch, obs_batch, valid_action_batch, last_action_batch,\
                action_batch, \
                next_state_batch, next_obs_batch, next_valid_action_batch,\
                reward_batch, terminated_batch,\
                episode_len

    def train(self, episode_batches):
        s_batch, o_batch, va_batch, la_batch, a_batch, ns_batch, no_batch, nva_batch, r_batch, t_batch,episode_len = self.get_data(episode_batches)
        '''
        s (batch_size, seq_len, state_dim)
        o (batch_size, seq_len, n_agent , obs_dim)
        a (batch_size, seq_len, n_agent , action_dim)
        va (batch_size, seq_len, n_agent , n_actions)
        la (batch_size, seq_len, n_agent , action_dim)    
        '''
        s_batch = torch.FloatTensor(s_batch).to(self.device) # (batch_size, seq_len, state_dim)
        ns_batch = torch.FloatTensor(ns_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device) # (batch_size, seq_len, n_agent 1)
        t_batch = torch.FloatTensor(t_batch).to(self.device) # (batch_size, seq_len, n_agent 1)
        a_batch_numpy = copy.deepcopy(a_batch)
        a_batch = torch.LongTensor(a_batch).to(self.device) # (batch_size, seq_len, n_agent action_dim)
        nva_batch = torch.FloatTensor(nva_batch).to(self.device) # (batch_size, seq_len, n_agent n_actions)
        batch_size = g_args("batch_size") 
        self.init_hidden_states(batch_size)
        #计算 q values
        multiagent_out = []
        for t in range (episode_len):
            obs = o_batch[:,t] # (batch_size, n_agent, obs_dim)
            obs = np.concatenate(obs, axis=0) # (batch_size* n_agent, action_dim)
            last_action = la_batch[:,t] # (batch_size, n_agent, action_dim)
            last_action = np.concatenate(last_action, axis=0) # (batch_size* n_agent, action_dim)
            obs = torch.FloatTensor(obs).to(self.device)
            last_action = torch.FloatTensor(last_action).to(self.device)
            input = torch.cat( (obs, last_action),- 1).to(self.device )# (batch_size * n_agent, obs_dim + action_dim)
            action_values ,self.hidden_states = self.agent(input, self.hidden_states)
            action_values = action_values.view(batch_size, self.n_agents, -1)
            multiagent_out.append(action_values)
        multiagent_out = torch.stack(multiagent_out, dim = 1) # (batch_size, seq_len, n_agent, action_dim)
        chosen_action_qvals = torch.gather(multiagent_out, dim=3, index=a_batch).squeeze(3)
        # 计算 next q vlaues
        target_multiagent_out = []
        for t in range (episode_len):
            next_obs = no_batch [:,t] # (batch_size, n_agent, obs_dim)
            next_obs = np.concatenate(next_obs, axis=0)
            last_action = a_batch_numpy[:,t] # (batch_size, n_agent, action_dim)
            last_action = np.concatenate(last_action, axis=0)
            next_obs = torch.FloatTensor(next_obs).to(self.device)
            last_action = torch.FloatTensor(last_action).to(self.device)
            input = torch.cat( (next_obs , last_action),-1).to(self.device)
            action_values ,self.target_hidden_states = self.target_agent (input, self.target_hidden_states)
            action_values = action_values.view(batch_size, self.n_agents, -1)
            target_multiagent_out.append(action_values)
        target_multiagent_out = torch.stack(target_multiagent_out, dim = 1) # (batch_size, seq_len, n_agent, action_dim)    
        target_multiagent_out[nva_batch == 0] = -9999999  
        target_max_qvals = target_multiagent_out.max(dim=3)[0]
        #q_tot
        chosen_action_qvals = self.mixer(chosen_action_qvals,s_batch)
        target_max_qvals = self.target_mixer(target_max_qvals,ns_batch)
        #q_tot_target
        yi = r_batch + self.gamma * (1 - t_batch) * target_max_qvals
        #td_error
        loss = ((chosen_action_qvals - yi.detach()) ** 2 ).sum()/ yi.numel()
        #更新网络参数
        print('loss:', loss)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 10)
        self.optimizer.step()
        return loss.item()

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def init_hidden_states(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.target_hidden_states = self.target_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def on_reset(self,batch_size):
        self.init_hidden_states(batch_size)

    def save_models(self,path):
        if not os.path.exists(path): 
            os.makedirs(path)
        #save Agents
        torch.save(self.agent.state_dict(),path+"/agent.pth")
        #save Mixer Network
        torch.save(self.mixer.state_dict(),path+"/mixer.pth")
        #save optimizer
        torch.save(self.optimizer.state_dict(),path+"/opt.pth")

    def load_models(self,path):
        #相对路径
        #load Agents
        self.agent.load_state_dict(torch.load(f"{path}/agent.pth", map_location=lambda storage, loc: storage))
        #load Mixer Network
        # Load all tensors onto GPU 1
        self.mixer.load_state_dict(torch.load(f"{path}/mixer.pth", map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load(f"{path}/opt.pth", map_location=lambda storage, loc: storage))
        self.update_targets()

    def load_dict(self, dict):
        if not dict:
            return
        self.agent.load_state_dict(dict["agent"])
        self.mixer.load_state_dict(dict["mixer"])
        self.optimizer.load_state_dict(dict["opt"])
        self.update_targets()

    def save_dict(self):
        d = {}
        d["agent"] = self.agent.state_dict()
        d["mixer"] = self.mixer.state_dict()
        d["optimizer"] = self.optimizer.state_dict()
        return d
