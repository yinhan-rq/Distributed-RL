import numpy as np
import torch
import torch.nn as nn
from config import global_args as g_args
import torch.nn.functional as F

class GAIL():
    def __init__(self,Discrim,device=torch.device("cpu")):
        self.device = g_args("device")
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        self.lr = g_args("lr")
        self.D_lr = g_args("D_lr")
        self.gail_batch_size = g_args("gail_batch_size")
        self.gail_exp_batch_size = g_args("gail_exp_batch_size")
        self.gail_epoch = g_args("gail_epoch")
        self.imi_reward_weight = g_args("imi_reward_weight")
        self.episode_length = g_args("episode_length")
        self.thread_num = g_args("thread_num")
        self.reward_clip = g_args("reward_clip")
        self.expert_loss_weight = g_args("expert_loss_weight")
        self.D = Discrim
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.D_lr)   

    def train(self, buffer, IL_buffer):
        '''
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param IL_buffer:(ExpertBuffer) buffer containing the expert data, non-empty when use_gail
        '''
        train_info = {}
        for _ in range(self.gail_epoch):
            state_batch, next_state_batch = buffer.sample_state(self.gail_batch_size)
            state_batch = torch.from_numpy(state_batch)
            next_state_batch = torch.from_numpy(next_state_batch)
            state_exp_batch, next_state_exp_batch = IL_buffer.sample(self.gail_exp_batch_size)                
            logits_pi = self.D(state_batch, next_state_batch)
            logits_exp = self.D(state_exp_batch, next_state_exp_batch)
            #compute loss
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + self.expert_loss_weight * loss_exp #防止过拟合，乘一个loss weight
            self.D_optimizer.zero_grad()
            #backward
            loss_disc.backward()
            self.D_optimizer.step()

        #for eval
        with torch.no_grad():
            state_batch, next_state_batch = buffer.sample_state(self.gail_batch_size)
            state_exp_batch, next_state_exp_batch = IL_buffer.sample(self.gail_exp_batch_size) 
            logits_pi = self.D(state_batch, next_state_batch)
            logits_exp = self.D(state_exp_batch, next_state_exp_batch)
            for i in range(g_args("left_agent_num")):
                train_info["acc_pi_"+ str(i)] = (logits_pi < 0).float().mean(dim=0)[i].item()
            train_info["acc_exp"] = (logits_exp > 0).float().mean().item()


        #update reward in the batch   
        imitation_rewards = self.D.calculate_reward(buffer.obs[:-1], buffer.private_obs[1:])
        IL_rewards = imitation_rewards.detach().cpu().numpy()
        IL_rewards = self.imi_reward_weight * IL_rewards
        IL_rewards = np.clip(self.imi_reward_weight * IL_rewards, -self.reward_clip, self.reward_clip)
        buffer.rewards = buffer.rewards + IL_rewards
        return train_info
    

