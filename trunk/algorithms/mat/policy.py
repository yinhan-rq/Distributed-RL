import torch
import numpy as np
from config import global_args as g_args
from utils.util import update_linear_schedule
from utils.util import get_shape_from_obs_space
from algorithms.mat.transformer import MultiAgentTransformer


class TransformerPolicy:
    def __init__(self, obs_space, cent_obs_space, act_space):
        self.device = torch.device(g_args("device"))
        self.lr = g_args("lr")
        self.opti_eps = g_args("opti_eps")
        self.weight_decay = g_args("weight_decay")
        self.action_type = 'Discrete'
        self.thread_num = g_args("thread_num")
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        self.act_dim = act_space.n
        self.act_num = 1
        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = g_args("left_agent_num")
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.transformer = MultiAgentTransformer(self.share_obs_dim, self.obs_dim, self.act_dim, self.num_agents)
        # count the volume of parameters of model
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0
        # for param in self.transformer.parameters():
        #     mulValue = np.prod(param.size())
        #     Total_params += mulValue
        #     if param.requires_grad:
        #         Trainable_params += mulValue
        #     else:
        #         NonTrainable_params += mulValue
        # print(f'Total params: {Total_params}')
        # print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        actions, action_probs, values = self.transformer.get_actions(obs, available_actions, deterministic)

        actions = actions.view(-1, self.act_num)
        action_probs = action_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        # unused, just for compatibility
        return values, actions, action_probs 

    def get_values(self, obs):
        """
        Get value function predictions.
        """
        _pre_process = lambda o: np.array(np.split(_t2n(o), self.thread_num))
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        values = self.transformer.get_values(obs)
        values = values.view(-1, 1)
        values = _pre_process(values)
        return values

    def evaluate_actions(self, obs, actions, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        """
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        action_log_probs, values, entropy = self.transformer(obs, actions, available_actions)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy

    def act(self, obs, available_actions=None, deterministic=True):
        """
        Compute actions using the given inputs.
        """
        _, actions, actions_probs = self.get_actions(obs, available_actions, deterministic)
        return actions, actions_probs

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

def _t2n(x):
    return x.detach().cpu().numpy()
