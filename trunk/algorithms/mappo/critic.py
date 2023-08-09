import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space,prettyprint
from config import global_args as g_args

class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    """
    def __init__(self, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = g_args("hidden_size")
        self._use_orthogonal = g_args("use_orthogonal")
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs):
        """
        Compute actions from the given inputs.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)
        return values
