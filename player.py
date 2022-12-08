import torch as th
import random
from utils import *
import torch.nn as nn
from move import *


class Player:
    '''This is the interface between the actor and the game.'''

    def __init__(self, ID: int, actor : type[nn.Module], influences=[None, None]):#, game):

        self.coins = 2
        self.influence = influences
        self.influence_alive = [True,True]
        self.alive = True
        
        # player ID is important for target masking in declare action
        self.player_id = ID
        
#         self.game = game
        # add self.influence_to_keep
        self.influence_to_keep = 0
        self.actor = actor

        # action prob histories


    def lose_coins(self, requested_coins):
        '''
        Lose coins (up to `requested_coins`) and return the amount lost.
        This function ensures that the player never has a negative amount of coins.
        '''
        num_lost = min(self.coins, requested_coins)
        self.coins -= num_lost
        return num_lost
    

    
    def update_influence_to_keep(self, influence_to_keep_dist):
        '''
        this updates the player's preference for which influence they'd prefer to keep,
        in the event that they have to give up an influence. 
        '''
        
        if self.influence_alive[0]:
            self.influence_to_keep = 1
            return
        elif not self.influence_alive[1]:
            self.influence_to_keep = 0
            return
        else:
            influence_to_keep_choice = th.multinomial(influence_to_keep_dist,1)
            assert(influence_to_keep_choice == AV_KEEP_CARD_0 or influence_to_keep_choice == AV_KEEP_CARD_1)
            self.influence_to_keep = influence_to_keep_choice - AV_KEEP_CARD_0


            
    def receive_card(self, card):
        ... # TODO

    
    def lose_influence(self):
        '''Player loses their least prefered influence'''
        idx = 1 - self.influence_to_keep
        influence = self.influence[idx]
        self.influence_alive[idx] = False
        # self.influence[idx] = None
        if sum(self.influence_alive) ==0:
            self.alive = False
        return influence
    

    
    
    

            
    ## we don't need a blocking method because all blocks are the same and are handled by the game
    # def block(self, action):
    #     # return None
    #     return blocking_card


    def mask_gsv(self,game_state):
        # mask the game state based on what players should or should not know.
        mask = th.zeros(game_state.shape)
        mask[:] = NOT_MY_TURN_MASK
        acting_turns = game_state[:,GSV_ACTING_PLAYER_0 + self.player_id] == 1
        mask[acting_turns, :] += WAS_MY_ACTION_MASK
        blocking_turns = game_state[:,GSV_TARGET_PLAYER_0 + self.player_id] == 1
        mask[blocking_turns,:] += WAS_MY_BLOCK_MASK
        game_state = game_state*mask


        #ensure that every player thinks they are player 0

        permutation = PLAYER_ORDERING_PERMUTATIONS[self.player_id]
        tensor_to_perm = game_state[:,GSV_ACTING_PLAYER_0:GSV_ACTING_PLAYER_4+1]
        tensor_to_perm = tensor_to_perm[:,permutation]
        game_state[:,GSV_ACTING_PLAYER_0:GSV_ACTING_PLAYER_4+1] = tensor_to_perm

        tensor_to_perm = game_state[:, GSV_TARGET_PLAYER_0:GSV_TARGET_PLAYER_4 + 1]
        tensor_to_perm = tensor_to_perm[:,permutation]
        game_state[:, GSV_TARGET_PLAYER_0:GSV_TARGET_PLAYER_4 + 1] = tensor_to_perm[:, permutation]

        tensor_to_perm = game_state[:, GSV_CHALLENGE_BY_PLAYER_0:GSV_CHALLENGE_BY_PLAYER_4 + 1]
        tensor_to_perm = tensor_to_perm[:,permutation]
        game_state[:, GSV_CHALLENGE_BY_PLAYER_0:GSV_CHALLENGE_BY_PLAYER_4 + 1] = tensor_to_perm[:, permutation]

        tensor_to_perm = game_state[:, GSV_CHALLENGE_BLOCK_BY_PLAYER_0:GSV_CHALLENGE_BLOCK_BY_PLAYER_4 + 1]
        tensor_to_perm = tensor_to_perm[:,permutation]
        game_state[:, GSV_CHALLENGE_BLOCK_BY_PLAYER_0:GSV_CHALLENGE_BLOCK_BY_PLAYER_4 + 1] = tensor_to_perm[:, permutation]
        #Check the initialization token to see that the reordering is correct
        assert(game_state[self.player_id, GSV_ACTING_PLAYER_0] == 1)
        return game_state

    def declare_action(self, player_list: list, game_state):
        '''Declare what action the player is going to take'''

        #Make Action Mask
        action_mask = th.zeros(ACTION_VEC_LEN)
        if self.coins >= 10:
            action_mask[AV_COUP] = 1
        else:
            action_mask[AV_INCOME] = 1
            action_mask[AV_FOREIGN_AID] = 1
            if self.coins >= 3:
                action_mask[AV_ASSASSINATE] = 1
            elif self.coins >= 7:
                action_mask[AV_COUP] = 1
            action_mask[AV_TAX] = 1
            action_mask[AV_EXCHANGE] = 1
            action_mask[AV_STEAL] = 1

        # Make target mask
        target_mask = th.zeros(ACTION_VEC_LEN)
        for player in player_list:
            if player.alive and player is not self:
                # the < section is to account for the fact that every player thinks they are player_id 0
                # as far as the actor is concerned. Which is why target_mask only contains player1 through 4.
                target_mask[AV_TARGET_PLAYER_1+player.player_id+(player.player_id < self.player_id)] = 1
        game_state = self.mask_gsv(game_state)
        action_dist, influence_to_keep_dist, target_dist, exchange_dist = self.actor(game_state,action_mask,target_mask=target_mask)
        
        self.update_influence_to_keep(influence_to_keep_dist)
        action_choice = th.multinomial(action_dist,1)
        #choose target
        target_choice = th.multinomial(target_dist,1)
        appropriate_permutation = PLAYER_ORDERING_PERMUTATIONS[self.player_id]
        player_index = appropriate_permutation[target_choice-AV_TARGET_PLAYER_1]
        assert(player_index != self.player_id)
        true_order_player_list = [None]*5
        for player in player_list:
            true_order_player_list[player.player_id] = player
        target = true_order_player_list[player_index]
        assert(target != self)
        assert(target.alive)

        if action_choice == AV_INCOME:
            action = Income(self)


        elif action_choice == AV_FOREIGN_AID:
            action =  Foreign_Aid(self)

        elif action_choice == AV_COUP:
            action =  Coup(self,target)

        elif action_choice == AV_TAX:
            action =  Tax(self)

        elif action_choice == AV_ASSASSINATE:
            action =  Assassinate(self,target)

        elif action_choice == AV_STEAL:
            action =  Steal(self,target)

        elif action_choice == AV_EXCHANGE:
            action =  Exchange(self,target,exchange_dist)
        
        else:
            raise ValueError(f"invalid Action Vector choice of {action_choice} in declare action")

        return action

    #Include the action in declare block for masking purposes
    def declare_block(self,action, game_state):
        block_mask = th.zeros(ACTION_VEC_LEN)
        block_mask[0] = 1
        action_type = type(action)
        if action_type ==  Foreign_Aid:
            block_mask[AV_BLOCK_FOREIGN_AID] = 1
        elif action.target == self:
            if action_type == Assassinate:
                block_mask[AV_BLOCK_ASSASSINATION] = 1
            elif action_type == Steal:
                block_mask[AV_BLOCK_STEALING_AMBS] = 1
                block_mask[AV_BLOCK_STEALING_CPT] = 1
        
        block_dist, influence_to_keep_dist, _, _ = self.actor(game_state,block_mask)
        block_choice = th.multinomial(block_dist,1)
        if(block_choice == AV_NOOP ):
            block = None
        elif(block_choice == AV_BLOCK_FOREIGN_AID):
            # Note sure what to return here yet
            block = Block(self,action,'Duke')
        elif(block_choice == AV_BLOCK_STEALING_CPT):
            block = Block(self,action,'Captain')
        elif(block_choice == AV_BLOCK_STEALING_AMBS):
            block = Block(self,action,'Ambassador')
        elif(block_choice == AV_BLOCK_ASSASSINATION):
            block = Block(self,action,'Contessa')
        else:
            raise ValueError(f"invalid Action Vector selection of {block_choice} in declare_block")

            

        self.update_influence_to_keep(influence_to_keep_dist)

        return block

    def declare_challenge(self,action,game_state):
        if action.character == None:
            return False
        challenge_mask = th.zeros(ACTION_VEC_LEN)
        challenge_mask[AV_CHALLENGE] = 1
        challenge_dist,influence_to_keep_dist,_,_ = self.actor(game_state,challenge_mask)

        challenge_choice = single_sample(challenge_dist*1)

        self.update_influence_to_keep(influence_to_keep_dist)
        return challenge_choice

    #This code is a bit repetetive compared to declare challenge, just has a different mask. 
    def declare_challenge_to_block(self,block, game_state):
        challenge_block_mask = th.zeros(ACTION_VEC_LEN)
        challenge_block_mask[AV_CHALLENGE_BLOCK] = 1
        challenge_block_dist,influence_to_keep_dist,_,_ = self.actor(game_state,challenge_block_mask)
        challenge_block_choice = single_sample(challenge_block_dist)
        self.update_influence_to_keep(influence_to_keep_dist)
        return challenge_block_choice

            



    # Start writing code here...