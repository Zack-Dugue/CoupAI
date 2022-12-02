import torch as th
import random
from utils import *
from move import *


class Player:
    '''This is the interface between the agent and the game.'''

    def __init__(self, influences: list, ID: int):#, game):

        self.coins = 2
        self.influence = influences
        self.influence_alive = (True,True)
        self.alive = True
        
        # player ID is important for target masking in declare action
        self.player_id = ID
        
#         self.game = game
        # add self.influence_to_keep
        self.influence_to_keep = 0



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
        this updates which card to keep
        '''
        # Huh? TODO
        
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

    def return_card(self, character: str):
        if self.influence[0] == character and self.influence_alive[0]:
            return 
            
    def receive_card(self, card):
        ... # TODO

    
    def lose_influence(self):
        '''Player loses their least prefered influence'''
        idx = 1 - self.influence_to_keep
        influence = self.influences[idx]
        self.influences[idx] = None
        return influence
    
    # TODO: game interface is different, make sure these agree
    def swap_influence(self, influence_idx: int,  deck):
        '''
        called after winning a challenge
        a player swaps either their 0th or 1st influence (based on character) with a card in the deck
        '''   
        assert influence_idx==0 or influence_idx==1, f'influence_idx is invalid (must be 0 or 1 but is {influence_idx})'
        character = self.influence[influence_idx]
        deck.append(character)
        random.shuffle(deck)
        new_influence = deck.pop(0)
        self.influence[influence_idx] = new_influence
    
    
    

            
    ## we don't need a blocking method because all blocks are the same and are handled by the game
    # def block(self, action):
    #     # return None
    #     return blocking_card
    
    def declare_action(self, player_list: list, game_state):
    
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
        player_list.remove(player.id)
        for k,player in enumerate(player_list):
            if player.alive:
                target_mask[AV_TARGET_PLAYER_1+k] = 1
        
        action_dist, influence_to_keep_dist, target_dist, exchange_dist = self.agent(game_state)
        self.update_influence_to_keep(influence_to_keep_dist)
        action_choice = th.multinomial(action_dist,1)
        target_choice = th.multinomial(target_dist,1)
        target = player_list[target_choice-AV_TARGET_PLAYER_1]

        if action_choice == AV_INCOME:
            return Income(self)
        
        elif action_choice == AV_FOREIGN_AID:
            return Foreign_Aid(self);
        
        elif action_choice == AV_COUP:
            return Coup(self,target)
        
        elif action_choice == AV_TAX:
            return Tax(self)
        
        elif action_choice == AV_ASSASSINATE: 
            return Assassinate(self,target)

        elif action_choice == AV_STEAL:
            return Steal(self,target)    

        elif action_choice == AV_EXCHANGE:
            return Exchange(self,exchange_dist)
        
        else:
            raise ValueError(f"invalid Action Vector choice of {action_choice} in declare action")
            
    #Include the action in declare block for masking purposes
    def declare_block(self,action, game_state):
        block_mask = th.zeros(ACTION_VEC_LEN)
        block_mask[0] = 1
        action_type = type(action)
        if action_type == ForeignAid:
            block_mask[AV_BLOCK_FOREIGN_AID] = 1
        elif action.target == self:
            if action_type == Assasinate:
                block_mask[AV_BLOCK_ASSASSINATION] = 1
            elif action_type == Steal:
                block_mask[AV_BLOCK_STEALING_AMBS] = 1
                block_mask[AV_BLOCK_STEALING_CPT] = 1
        
        block_dist, influence_to_keep_dist, _, _ = self.agent(game_state,block_mask)
        block_choice = th.multinomial(block_dist,1)
        if(block_choice == AV_NOOP ):
            block = None
        elif(block_choice == AV_BLOCK_FOREIGN_AID):
            # Note sure what to return here yet
            block = Block(self,'Duke')
        elif(block_choice == AV_BLOCK_STEALING_CPT):
            block = Block(self,'Captain')
        elif(block_choice == AV_BLOCK_STEALING_AMBS):
            block = Block(self,'Ambassador')
        elif(block_choice == AV_BLOCK_ASSASSINATION):
            block = Block(self,'Contessa')
        else:
            raise ValueError(f"invalid Action Vector selection of {block_choice} in declare_block")

            

        self.update_influence_to_keep(influence_to_keep_dist)

        return block

    def declare_challenge(self,game_state):
        challenge_mask = th.zeros(ACTION_VEC_LEN)
        challenge_mask[AV_CHALLENGE] = 1
        challenge_dist,influence_to_keep_dist,_,_ = self.agent(game_state,challenge_mask)
        challenge_choice = th.multinomial(challenge_dist,1)

        self.update_influence_to_keep(influence_to_keep_dist)
        if challenge_choice == AV_NOOP:
            return False
        elif challenge_choice == AV_CHALLENGE:
            return True
        else:
            print(f"Something has gone horribly wrong in declare challenge. Invalid Action Vector selection of: {challenge_choice}")
            assert(False)

    #This code is a bit repetetive compared to declare challenge, just has a different mask. 
    def declare_challenge_to_block(self,block, game_state):
        challenge_block_mask = th.zeros(ACTION_VEC_LEN)
        challenge_block_mask[AV_CHALLENGE_BLOCK] = 1
        challenge_block_dist,influence_to_keep_dist,_,_ = self.agent(game_state,challenge_block_mask)
        challenge_block_choice = th.multinomial(challenge_block_dist,1)
        self.update_influence_to_keep(influence_to_keep_dist)
        if challenge_block_choice == AV_NOOP:
            return False
        elif challenge_block_choice == AV_CHALLENGE_BLOCK:
            return True
        else:
            print(f"Something has gone horribly wrong in declare challenge_block. Invalid Action Vector selection of: {challenge_block_choice}")
            assert(False);          
            
            



    # Start writing code here...