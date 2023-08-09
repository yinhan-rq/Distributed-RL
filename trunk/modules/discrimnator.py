import torch
from torch import nn
import torch.nn.functional as F
from algorithms.utils.mlp import MLPBase
from config import global_args as g_args
from utils.util import get_shape_from_obs_space
from algorithms.utils.util import init,check

class GAILDiscrim(nn.Module):

    def __init__(self, obs_space, private_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = g_args("hidden_size")
        self._use_orthogonal = g_args("use_orthogonal")
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        obs_shape = get_shape_from_obs_space(obs_space)
        pri_obs_shape = get_shape_from_obs_space(private_obs_space)
        #s + s'
        base_shape = list(obs_shape)
        pri_obs_shape = list(pri_obs_shape)
        base_shape[0] = base_shape[0] + pri_obs_shape[0]
        self.base = MLPBase(base_shape)
        self.D_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, state, next_state):
        state = check(state).to(**self.tpdv)
        next_state = check(next_state).to(**self.tpdv)
        
        D_features = self.base(torch.cat([state, next_state], dim = -1))
        return self.D_out(D_features)

    def calculate_reward(self, state, next_state):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)],which is a bonus terms
        #addition penalty from DAC paper: E_{\pi} [log(D)]
        # DISCRIMINATOR-ACTOR-CRITIC:ADDRESSING SAMPLE INEFFICIENCY AND REWARD BIAS IN ADVERSARIAL IMITATION LEARNING(2018)
        #so the final reward is : R = E_{\pi} [-log(1 - D)] + E_{\pi} [log(D)]
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(state, next_state)) + F.logsigmoid(self.forward(state, next_state))
