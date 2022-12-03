import torch as th
from torch import nn
from torch import functional as F

card_to_keep_mask = th.zeros(31)
card_to_keep_mask[19] = 1
card_to_keep_mask[20] = 1
exchange_mask = th.zeros(31)
exchange_mask[21:] = 1


class GEGelU(nn.Module):
    def __init__(self,dim):
        super(GEGelU, self).__init__()
        self.dim = dim
        self.activation = nn.GELU()
    def forward(self,input):
        return input[:self.dim]*self.activation(input[self.dim:])


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('layer_1 transformer', nn.TransformerEncoderLayer(68,8))
        self.model.add_module('layer_2 transformer', nn.TransformerEncoderLayer(68,2))
        self.model.add_module('layer_3 transformer', nn.TransformerEncoderLayer(68,2))
        self.model.add_module('layer_4 transformer', nn.TransformerEncoderLayer(68,2))
        self.model.add_module('layer_5 transformer', nn.TransformerEncoderLayer(68,2))
        self.model.add_module('avg_pooling', nn.AvgPool1d(68))
        self.model.add_module('FC1', nn.Linear(68,4096))
        self.model.add_module('non_linearity_1', GEGelU(2048))
        self.model.add_module('FC2', nn.Linear(68,4096))
        self.model.add_module('non_linearity_2', GEGelU(2048))
        self.model.add_module('FC3', nn.Linear(2048,31))
        self.target_mask = th.zeros(31)
        self.target_mask[7:13] = 1
        self.card_to_keep_mask = th.zeros(31)
        self.card_to_keep_mask[18:20] = 1

    def forward(self, game_state = th.Tensor, action_mask = th.tensor):
        """returns 3 different tensors:
         --a distribution over which cards to keep
         --a distribution over which targets
         --a distribution over actions"""
        action_state = self.model.forward(game_state)
        targets = th.softmax(self.target_mask*action_state,1)
        cards_to_keep = th.softmax(action_state,1)
        actions = th.softmax(action_mask*action_state,1)
        exchange_dist = th.softmax(action_state*exchange_mask,1)

        return actions, targets, cards_to_keep, exchange_dist

class RandomAgent(nn.Module):
    def __init__(self):
        super(RandomAgent,self).__init__()


    def forward(self, game_state : th.Tensor, action_mask: th.Tensor, target_mask=th.zeros(30)):
        action_state =  th.ones(30)
        actions = th.softmax(action_state * action_mask,1)
        cards_to_keep = th.softmax(action_state*card_to_keep_mask,1)
        exchange_dist = th.softmax(action_state*exchange_mask,1)
        targets = th.softmax(action_state*target_mask,1)
        return actions,cards_to_keep,targets,exchange_dist  

