import torch as th
from utils import *
from torch import nn
from torch import functional as F

card_to_keep_mask = th.zeros(ACTION_VEC_LEN)
card_to_keep_mask[AV_KEEP_CARD_0] = 1
card_to_keep_mask[AV_KEEP_CARD_1] = 1
exchange_mask = th.zeros(ACTION_VEC_LEN)
exchange_mask[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_CONTESSA + 1] = 1

class SofterMax(nn.Module):
    def __init__(self):
        '''Softer Max first masks the input x, 
        and then performs a softmax operation only over the (un)masked region.
        If there is only one unmasked element, it applies a sigmoid over that element.
        
        SofterMax returns the same shape as the input'''
        super(SofterMax, self).__init__()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,mask):

        mask_inds = mask > 0
        x = x*mask
        if sum(mask_inds) == 1:
            x[mask_inds] = self.sigmoid(x[mask_inds])
        else:
            x[mask_inds] = self.softmax(x[mask_inds])
        return x

class GEGelU(nn.Module):
    def __init__(self,dim):
        super(GEGelU, self).__init__()
        self.dim = dim
        self.activation = nn.GELU()
    def forward(self,input):
        return input[:,:self.dim]*self.activation(input[:,self.dim:])

class MyPool(nn.Module):
    def __init__(self,dim,length):
        super(MyPool, self).__init__()
        self.dim = dim
        self.length = length
    def forward(self, input):
        return th.sum(input, self.dim)/self.length

class LearnedQueryAttention(nn.Module):
    '''a potential replacement for avg pool at the end
    that uses learned query attention to extract the features which get passed
    on to the end feed forward part'''
    #WORK IN PROGRESS
    def __init__(self, dim,num_queries):
        super(LearnedQueryAttention, self).__init__()
        self.learned_queries = th.random(num_queries,dim)

    def forward(self,input):
        attention = th.matmul(input,self.learned_queries)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.softermax = SofterMax()
        self.model = nn.Sequential()
        self.model.add_module('layer_1 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,8,dropout=0))
        self.model.add_module('layer_2 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,8,dropout=0))
        self.model.add_module('layer_3 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,4,dropout=0))
        self.model.add_module('layer_4 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,4,dropout=0))
        self.model.add_module('layer_5 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,2,dropout=0))
        self.model.add_module('layer_6 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,2,dropout=0))
        self.model.add_module('layer_7 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,1,dropout=0))
        self.model.add_module('layer_8 transformer', nn.TransformerEncoderLayer(GAME_STATE_VEC_LEN,1,dropout=0))

        self.model.add_module('avg_pooling',  MyPool(1,GAME_STATE_VEC_LEN))
        self.model.add_module('FC1', nn.Linear(GAME_STATE_VEC_LEN,4096))
        self.model.add_module('non_linearity_1', GEGelU(2048))
        self.model.add_module('FC2', nn.Linear(2048,4096))
        self.model.add_module('non_linearity_2', GEGelU(2048))
        self.model.add_module('FC3', nn.Linear(2048,ACTION_VEC_LEN))
        self.target_mask = th.zeros(31)
        self.target_mask[7:13] = 1
        self.card_to_keep_mask = th.zeros(31)
        self.card_to_keep_mask[18:20] = 1

    def forward(self, game_state : th.Tensor, action_mask: th.Tensor, target_mask=th.zeros(ACTION_VEC_LEN)):
        """returns 3 different tensors:
         --a distribution over which cards to keep
         --a distribution over which targets
         --a distribution over actions"""
        action_state = self.model.forward(game_state.view([1,game_state.shape[0],game_state.shape[1]]))
        actions = self.softermax(action_state[0],action_mask)
        cards_to_keep = self.softermax(action_state[0],card_to_keep_mask)
        exchange_dist = self.softermax(action_state[0],exchange_mask)
        targets = self.softermax(action_state[0],target_mask)
        return actions,cards_to_keep,targets,exchange_dist



class RandomAgent(nn.Module):
    def __init__(self):
        super(RandomAgent,self).__init__()
        self.softermax = SofterMax()


    def forward(self, game_state : th.Tensor, action_mask: th.Tensor, target_mask=th.zeros(30)):
        action_state =  th.ones(ACTION_VEC_LEN)
        actions = self.softermax(action_state,action_mask)
        cards_to_keep = self.softermax(action_state,card_to_keep_mask)
        exchange_dist = self.softermax(action_state,exchange_mask)
        targets = self.softermax(action_state,target_mask)
        return actions,cards_to_keep,targets,exchange_dist  

