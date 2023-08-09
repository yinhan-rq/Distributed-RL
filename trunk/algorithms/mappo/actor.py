import torch
import torch.nn as nn
from algorithms.utils.util import check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space
from config import global_args as g_args


class R_Actor(nn.Module):
    """
    """
    def __init__(self, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = g_args("hidden_size")
        self._gain = g_args("gain")
        self._use_orthogonal = g_args("use_orthogonal")
        
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(obs_shape)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        """
        _to_tensor = lambda o : check(o).to(**self.tpdv) if o is not None else o
        obs = _to_tensor(obs)
        available_actions = _to_tensor(available_actions)
        actor_features = self.base(obs)

        actions, action_probs = self.act(actor_features, available_actions, deterministic)
        return actions, action_probs

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        _to_tensor = lambda o : check(o).to(**self.tpdv) if o is not None else o
        obs = _to_tensor(obs)
        action = _to_tensor(action)
        available_actions = _to_tensor(available_actions)
        actor_features = self.base(obs)

        action_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks)
        return action_probs, dist_entropy

