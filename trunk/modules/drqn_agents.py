
import torch
import numpy as np
from rnn_network import RNN




# This multi-agent controller shares parameters between agents
class MultiAgent:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        #input shape : obs+last action
        input_shape = int(np.prod(args.state_shape))+args.action_dim #足球默认1
        # shares parameters between agents' networks
        self.network = RNN(input_shape,args)
        self.hidden_state = None
        


    def selection_actions(ep_batch,):
        pass

    def forward(ep_batch,test_mode=False):
        pass

    def build_inputs(self, batch, time):
        pass

    def init_hidden(self, batch_size):
        #  rnn的hidden_state=[1,hidden_num]
        #  这里为[batch_size,agents_num,hidden_num]
        self.hidden_states = self.network.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    
    def parameters(self):
        return self.network.parameters()

    def load_state(self, agent):
        self.network.load_state_dict(agent.network.state_dict())

    def cuda(self):
        self.network.cuda()

    def save_models(self,path):
        torch.save(self.network.state_dict(), path+'agent.pth')
        
    def load_models(self,path):
        self.agent.load_state_dict(torch.load(path+'agent.pth'),map_location=lambda storage, loc: storage)

    