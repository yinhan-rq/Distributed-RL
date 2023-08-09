import copy
import torch
import numpy as np
import random
import os
from collections import deque
from config import global_args as g_args
from utils.util import get_shape_from_obs_space, get_shape_from_act_space


class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer_size = g_args("buffer_size")
        self.buffer = deque()

    def add(self, episode_data):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(episode_data)
        else:
            self.buffer.popleft()
            self.buffer.append(episode_data)

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size
    
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        random.sample(self.buffer, batch_size)
        batch = random.sample(self.buffer, batch_size)
        return batch

class ExpertBuffer:
    def __init__(self, buffer_size, agent_num, state_shape, single_state_shape, path="", device=torch.device("cpu")):
        self.device = device
        self.step = 0 
        self.n = 0
        self.buffer_size = buffer_size
        self.agent_num = agent_num
        self.states = torch.empty(
            (buffer_size, agent_num, *state_shape), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, agent_num, *single_state_shape), dtype=torch.float, device=device)            
        if path:
            self.load(path)

    def sample(self, batch_size):

        idxes = np.random.randint(low=0, high=self.n, size=batch_size)
        return (
            self.states[idxes],
            self.next_states[idxes]
        )

    def append(self, state, next_state):
        self.states[self.step].copy_(torch.from_numpy(state))
        self.next_states[self.step].copy_(torch.from_numpy(next_state))
        self.step = (self.step + 1) % self.buffer_size
        self.n = min(self.n + 1, self.buffer_size)
    
    def load(self,path):
        data = torch.load(path)
        self.states = data['states'].clone().to(self.device) 
        self.next_states = data["next_states"].clone().to(self.device)
        self.n = data['buffer_num']
        self.buffer_size = self.states.size()[0]

    def add_transition(self):
        pass

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'states': self.states.clone().cpu(),
            'next_states': self.next_states.clone().cpu(),
            'buffer_num': self.n
        }, path)

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    """

    def __init__(self, obs_space, cent_obs_space, pri_obs_space, act_space):
        self.episode_length = g_args("episode_length")
        self.n_rollout_threads = g_args("thread_num")
        self.hidden_size = g_args("hidden_size")
        self.recurrent_N = g_args("recurrent_N")
        self.gamma = g_args("gamma")
        self.gae_lambda = g_args("gae_lambda")
        self._use_gae = g_args("use_gae")
        num_agents = g_args("left_agent_num")
        obs_shape = get_shape_from_obs_space(obs_space)
        pri_obs_shape = get_shape_from_obs_space(pri_obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(pri_obs_shape[-1]) == list:
            pri_obs_shape = pri_obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        act_shape = get_shape_from_act_space(act_space)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),dtype=np.float32)
        self.private_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *pri_obs_shape), dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros( (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32 )
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),dtype=np.float32)
        else:
            self.available_actions = None

        self.step = 0

    def insert(self, data, masks, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        """
        data = copy.deepcopy(data)
        rewards = np.expand_dims(data["rewards"], -1)

        self.obs[self.step + 1] = data["obs"]
        self.share_obs[self.step + 1] = data["obs"]
        self.private_obs[self.step + 1] = data["private_obs"]
        self.actions[self.step] = data["actions"]
        self.action_log_probs[self.step] = data["action_log_probs"]
        self.value_preds[self.step] = data["values"]
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.private_obs[self.step + 1] = self.private_obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def get_obs(self, step):
        return np.concatenate(self.obs[step])

    def get_share_obs(self, step):
        return np.concatenate(self.share_obs[step])

    def get_available_actions(self, step):
        avail = np.array(self.obs[step][..., : 19])  #NOTE:动作空间固定19
        # return np.concatenate(avail)
        return avail

    def compute_returns(self, next_value):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            value_preds = self.value_preds[step]
            value_preds_next = self.value_preds[step + 1]

            delta = self.rewards[step] + self.gamma * self.masks[step + 1] * value_preds_next - value_preds
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            # delta = self.rewards[step] + self.gamma * value_preds_next - value_preds
            # gae = delta + self.gamma * self.gae_lambda * gae
            self.advantages[step] = gae
            self.returns[step] = gae + value_preds

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        sequence_obs = np.concatenate([self.obs[:-1], self.private_obs[1:]], axis=-1)
        sequence_obs = sequence_obs.reshape(-1, *sequence_obs.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)
        sample = {}
        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            sequence_obs_batch = sequence_obs[indices]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            obs_batch = sequence_obs_batch[..., : self.obs.shape[-1]]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            sample["share_obs_batch"] = share_obs_batch
            sample["obs_batch"] = obs_batch
            sample["actions_batch"] = actions_batch
            sample["available_actions_batch"] = available_actions_batch
            sample["active_masks_batch"] = active_masks_batch
            sample["value_preds_batch"] = value_preds_batch
            sample["return_batch"] = return_batch
            sample["masks_batch"] = masks_batch
            sample["old_action_log_probs_batch"] = old_action_log_probs_batch
            sample["adv_targ"] = adv_targ
            yield sample

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            sample = {}
            sample["share_obs_batch"] = share_obs_batch
            sample["obs_batch"] = obs_batch
            sample["actions_batch"] = actions_batch
            sample["available_actions_batch"] = available_actions_batch
            sample["active_masks_batch"] = active_masks_batch
            sample["value_preds_batch"] = value_preds_batch
            sample["return_batch"] = return_batch
            sample["old_action_log_probs_batch"] = old_action_log_probs_batch
            sample["adv_targ"] = adv_targ
            yield sample
       
    # this function is used for imitation learning when given the batch size but not the mini_batch number                  
    def sample_state(self, batch_size):
        indices = np.random.randint(low=0, high=self.episode_length, size=batch_size)
        sequence_obs = np.concatenate([self.obs[:-1], self.private_obs[1:]],axis = -1)
        sequence_obs = sequence_obs.reshape(-1, *sequence_obs.shape[2:])
        sequence_obs_batch = sequence_obs[indices]
        obs_batch = sequence_obs_batch[..., : self.obs.shape[-1]]
        next_obs_batch = sequence_obs_batch[..., self.obs.shape[-1]: self.obs.shape[-1] + self.private_obs.shape[-1]]     
        return obs_batch, next_obs_batch


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols