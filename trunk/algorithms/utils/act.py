import torch
import torch.nn as nn
from algorithms.utils.util import check
from .distributions import Categorical
from config import global_args as g_args

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        device = torch.device(g_args("device"))
        self.tpdv = dict(dtype=torch.float32, device=device)
        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
   
    def forward(self, x, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        """
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy
