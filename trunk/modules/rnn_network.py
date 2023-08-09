import torch.nn as nn
import torch.nn.functional as F
from config import global_args as g_args


class RNN(nn.Module):
    def __init__(self, input_shape):
        super(RNN, self).__init__()
        print('input_shape: ', input_shape)
        self.fc1 = nn.Linear(input_shape,  g_args("rnn_hidden_dim"))
        self.rnn = nn.GRUCell(g_args("rnn_hidden_dim"), g_args("rnn_hidden_dim"))
        self.fc2 = nn.Linear(g_args("rnn_hidden_dim"), g_args("n_actions"))
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, g_args("rnn_hidden_dim") ) .zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1,  g_args("rnn_hidden_dim") )
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
