from player import Player
from utils import *
import torch as th
import torch.nn as nn 
import torch.functional as F
import random

class Move:
    '''This is the super class for Block and Action.'''

    def __init__(self, player, character):
        self.player = player
        self.character = character

    def return_card(self):
        return self.player.return_card(self.character)

    def is_valid(self):
        '''Return True if `self.player` has the right character to play this move.'''
        assert self.character is not None, 'This move cannot be challenged'
        return self.player.has_character(self.character)

class Block(Move):
    
    def __init__(self, player, action, character):
        self.action = action
        super().__init__(player, character)

class Action(Move):
    '''This is the super class for all actions (i.e. assasinate, exchange, etc)'''

    def __init__(self, player: Player, target: Player, character: str, counter_characters: list):
        self.target = target
        self.counter_characters = counter_characters
        super().__init__(player, character)

    def incur_costs():
        '''
        Player incurs cost for playing action (i.e. losing coins). 
        This is left intentionall blank.
        '''
        ...

    def do_Action(self, deck : list):
        '''
        Carry out the action (i.e. target gets assasinated). 
        This is left intentionall blank.
        '''
        ...
        
# Specific Actions

        
class Income(Action):

    def __init__(self, player):
        super(Income,self).__init__(player, None, None,[])
        
    
    def do_Action(self, deck : list):
        self.player.coins += 2

class Foreign_Aid(Action):

    def __init__(self, player):
        super().__init__(player, None, None, ['Duke'])
    
    def do_Action(self, deck : list):
        self.player.coins += 2

class Coup(Action):   
    
    def __init__(self, player, target):
        assert player.coins >= 7, '`player` does not have enough coins for coup'
        super().__init__(player, target, None, [])
        
    def incur_costs(self):
        self.player.coins -= 7
    
    def do_Action(self, deck : list):
        self.target.lose_influence()

class Tax(Action):

    def __init__(self, player):
        super().__init__(player, None, 'Duke', [])

    def do_action(self, deck : list):
        self.player.coins += 3

class Exchange(Action):
   
    def __init__(self, player, target, influence_dist,deck):
        super().__init__(player, target, 'Ambassador', [])
    '''
    influence_dist is the distribution of
    '''
    # I feel like exchange should probably be handled in this class since all 
    # the other classes are pretty much self contained. 
    # This one edge case makes the whole action as a class thing kinda painful. Rip. 
    #TODO finish this: 
    def do_Action(self, deck : list):
        draw_0 = deck.pop(0)
        draw_1 = deck.pop(0)
        random.shuffle(deck)
        if(self.player.influence_alive[0]):
            self.influence_dist[AV_EXCHNG_CARD_0_AMBASSADOR] *= (draw_0 == 'Ambassador' or draw_1 == 'Ambassador' or self.influence[0] == 'Ambassador')
            self.influence_dist[AV_EXCHNG_CARD_0_DUKE] *= (draw_0 == 'Duke' or draw_1 == 'Duke' or self.influence[0] == 'Duke')
            self.influence_dist[AV_EXCHNG_CARD_0_ASSASSIN] *= (draw_0 == 'Assassin' or draw_1 == 'Assassin' or self.influence[0] == 'Assassin')
            self.influence_dist[AV_EXCHNG_CARD_0_CAPTAIN] *= (draw_0 == 'Captain' or draw_1 == 'Captain' or self.influence[0] == 'Captain')
            self.influence_dist[AV_EXCHNG_CARD_0_CONTESSA] *= (draw_0 == 'Contessa' or draw_1 == 'Contessa' or self.influence[0] == 'Contessa')
            self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_0_CONTESSA+1] = nn.softmax(self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_DUKE])
            new_card = th.multinomial(self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_0_CONTESSA+1],1)
            self.influence[0] = new_card
            if new_card == draw_0: draw_0 = None
            elif new_card == draw_1: draw_1 = None    
        if(self.player.influence_alive[1]):
            self.influence_dist[AV_EXCHNG_CARD_1_AMBASSADOR] *= (draw_0 == 'Ambassador' or draw_1 == 'Ambassador' or self.influence[1] == 'Ambassador')
            self.influence_dist[AV_EXCHNG_CARD_1_DUKE] *= (draw_0 == 'Duke' or draw_1 == 'Duke' or self.influence[1] == 'Duke')
            self.influence_dist[AV_EXCHNG_CARD_1_ASSASSIN] *= (draw_0 == 'Assassin' or draw_1 == 'Assassin' or self.influence[1] == 'Assassin')
            self.influence_dist[AV_EXCHNG_CARD_1_CAPTAIN] *= (draw_0 == 'Captain' or draw_1 == 'Captain' or self.influence[1] == 'Captain')
            self.influence_dist[AV_EXCHNG_CARD_1_CONTESSA] *= (draw_0 == 'Contessa' or draw_1 == 'Contessa' or self.influence[1] == 'Contessa')
            self.influence_dist[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA+1] = nn.softmax(self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_DUKE])
            new_card = th.multinomial(self.influence_dist[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA+1],1)
            self.influence[1] = new_card
                
        


class Assassinate(Action):

    def __init__(self, player, target):
        assert player.coins >= 3, '`player` does not have enough coins for assasination'
        super().__init__(player, target, 'Assasin', ['Contessa'])
        
    def incur_costs(self):
        self.player.coins -= 3
        
    def do_Action(self, deck : list):
        self.target.lose_influence()

class Steal(Action):

    def __init__(self, player, target):
        super().__init__(player, target, 'Captain', ['Ambassador', 'Captain'])
    
    def do_Action(self, deck : list):
        num_coins = self.target.lose_coins(2)
        self.player.coins += num_coins