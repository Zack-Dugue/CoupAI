from utils import *
import torch as th
import torch.nn as nn 
import torch.nn.functional as F
import random
from agent import SofterMax
## ALL refrences to player were removed
## Because of circular imports (move imports player and player imports move)
## Since player's import was only used for type checking purposes,
## It was determined okayer to remove. And the type checking feature was disabled
## (it should be implicitly checked via duck-typing anyways.

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
        return (( self.player.influence[0] == self.character and self.player.influence_alive[0]) or \
               (self.player.influence[1] == self.character and self.player.influence_alive[1]))


class Block(Move):
    
    def __init__(self, player, action, character):
        self.action = action
        super().__init__(player, character)

class Action(Move):
    '''This is the super class for all actions (i.e. assasinate, exchange, etc)'''

    def __init__(self, player , target, character: str, counter_characters: list, AV):
        self.target = target
        self.counter_characters = counter_characters
        self.AV = AV
        super().__init__(player, character)

    def incur_costs(self):
        '''
        Player incurs cost for playing action (i.e. losing coins). 
        This is left intentionall blank.
        '''
        ...

    def do_action(self, deck : list, game_state):
        '''
        Carry out the action (i.e. target gets assasinated).
        This is left intentionall blank.
        '''
        ...

# Specific Actions


class Income(Action):

    def __init__(self, player):
        super(Income,self).__init__(player, None, None,[], AV_INCOME)


    def do_action(self, deck : list, game_state):
        self.player.coins += 2

class Foreign_Aid(Action):

    def __init__(self, player):
        super().__init__(player, None, None, ['Duke'], AV_FOREIGN_AID)

    def do_action(self, deck : list, game_state):
        self.player.coins += 2

class Coup(Action):

    def __init__(self, player, target):
        assert player.coins >= 7, '`player` does not have enough coins for coup'
        super().__init__(player, target, None, [], AV_COUP)

    def incur_costs(self):
        self.player.coins -= 7

    def do_action(self, deck : list, game_state):
        self.target.lose_influence()

class Tax(Action):

    def __init__(self, player):
        super().__init__(player, None, 'Duke', [], AV_TAX)

    def do_action(self, deck : list, game_state):
        self.player.coins += 3


class Assassinate(Action):

    def __init__(self, player, target):
        assert player.coins >= 3, '`player` does not have enough coins for assasination'
        super().__init__(player, target, 'Assasin', ['Contessa'], AV_ASSASSINATE)

    def incur_costs(self):
        self.player.coins -= 3

    def do_action(self, deck : list, game_state):
        self.target.lose_influence()

class Exchange(Action):

    def __init__(self, player, target, influence_dist):
        '''
        influence_dist is the distribution of cards desired.
        '''
        super().__init__(player, None, 'Ambassador', [], AV_EXCHANGE)
        self.softermax = SofterMax()
        self.influence_dist = influence_dist

    def do_action(self, deck : list, game_state):
        
        #Draw both cards and update gamestate accordingly
        draw_0 = deck.pop(0)
        draw_1 = deck.pop(0)
        game_state[-1,GSV_EXCHANGE_CARD1_SEEN_DUKE + influence_to_num(draw_0)] = 1
        game_state[-1,GSV_EXCHANGE_CARD2_SEEN_DUKE + influence_to_num(draw_1)] = 1
        #record what influence the player used to have
        old_influence = self.player.influence.copy()
        mask = th.zeros(ACTION_VEC_LEN)
        if(self.player.influence_alive[0]):
            #TODO charollete make into forloop
            mask[AV_EXCHNG_CARD_0_DUKE] = (draw_0 == 'Duke' or draw_1 == 'Duke' or self.player.influence[0] == 'Duke')
            mask[AV_EXCHNG_CARD_0_ASSASSIN] = (draw_0 == 'Assassin' or draw_1 == 'Assassin' or self.player.influence[0] == 'Assassin')
            mask[AV_EXCHNG_CARD_0_AMBASSADOR] = (draw_0 == 'Ambassador' or draw_1 == 'Ambassador' or self.player.influence[0] == 'Ambassador')
            mask[AV_EXCHNG_CARD_0_CAPTAIN] = (draw_0 == 'Captain' or draw_1 == 'Captain' or self.player.influence[0] == 'Captain')
            mask[AV_EXCHNG_CARD_0_CONTESSA] = (draw_0 == 'Contessa' or draw_1 == 'Contessa' or self.player.influence[0] == 'Contessa')

            #Sample from influence_dist softermaxed over a mask influence[0] and the two cards drawn 
            self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_0_CONTESSA+1] = self.softermax(self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_DUKE],mask[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_DUKE])
            new_card = th.multinomial(self.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_0_CONTESSA+1],1)
            new_card = num_to_influnece(new_card)
            self.player.influence[0] = new_card
            #return influence to the deck if its been swapped for a draw card
            if new_card == draw_0:
                #draw_0=None so it can't be drawn again
                draw_0 = None
                deck.append(old_influence[0])
                game_state[-1,GSV_EXCHANGE_SWAP_1_1] = 1
            
            #return influence to the deck if its been swapped for a draw card
            elif new_card == draw_1:
                draw_1 = None
                deck.append(old_influence[0])
                game_state[-1,GSV_EXCHANGE_SWAP_1_2] =1

        mask = th.zeros(ACTION_VEC_LEN)
        if(self.player.influence_alive[1]):
            #TODO charollete make into forloop
            mask[AV_EXCHNG_CARD_1_DUKE] = (draw_0 == 'Duke' or draw_1 == 'Duke' or self.player.influence[1] == 'Duke')
            mask[AV_EXCHNG_CARD_1_ASSASSIN] = (draw_0 == 'Assassin' or draw_1 == 'Assassin' or self.player.influence[1] == 'Assassin')
            mask[AV_EXCHNG_CARD_1_AMBASSADOR] = (draw_0 == 'Ambassador' or draw_1 == 'Ambassador' or self.player.influence[1] == 'Ambassador')
            mask[AV_EXCHNG_CARD_1_CAPTAIN] = (draw_0 == 'Captain' or draw_1 == 'Captain' or self.player.influence[1] == 'Captain')
            mask[AV_EXCHNG_CARD_1_CONTESSA] = (draw_0 == 'Contessa' or draw_1 == 'Contessa' or self.player.influence[1] == 'Contessa')

            #Sample from influence_dist softermaxed over a mask influence[0] and the two cards drawn 
            self.influence_dist[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA+1] = self.softermax(self.influence_dist[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA + 1],mask[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA + 1])
            new_card = th.multinomial(self.influence_dist[AV_EXCHNG_CARD_1_DUKE:AV_EXCHNG_CARD_1_CONTESSA+1],1)
            new_card = num_to_influnece(new_card)
            self.player.influence[1] = new_card
            if new_card == draw_0:
                draw_0 = None
                game_state[-1,GSV_EXCHANGE_SWAP_2_1] = 1
                deck.append(old_influence[1])

            elif new_card == draw_1:
                draw_1 = None
                game_state[-1,GSV_EXCHANGE_SWAP_2_2] = 1
                deck.append(old_influence[1])
        
        # return any drawn card not taken by the character
        # back to the deck.
        if draw_0 is not None:
            deck.append(draw_0)
        if draw_1 is not None:
            deck.append(draw_1)
        random.shuffle(deck)




class Steal(Action):

    def __init__(self, player, target):
        super().__init__(player, target, 'Captain', ['Ambassador', 'Captain'], AV_STEAL)

    def do_action(self, deck : list, game_state):
        num_coins = self.target.lose_coins(2)
        self.player.coins += num_coins