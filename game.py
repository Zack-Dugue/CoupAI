import random
from utils import *
from move import *
from player import Player
#numpy is just for the Fourier Feature stuff
import numpy as np


# Start writing code here...


class Game:
    '''This is the interface between all the players.'''

    def __init__(self,players,deck):
        
        self.deck = deck
        self.players = players
        self.game_state_init(players)
        self.num_alive = len(self.players) # TODO


    def game_state_init(self,players):
        game_state = th.zeros([5,GAME_STATE_VEC_LEN])
        for player in players:
            id = player.player_id
            game_state[id,GSV_ACTING_PLAYER_0 + id] = 1
            game_state[id,GSV_ACTION_EXCHANGE] = 1
            influence_0 = player.influence[0]
            influence_1 = player.influence[1]
            game_state[id,GSV_EXCHANGE_CARD1_SEEN_DUKE + influence_to_num(influence_0)] = 1
            game_state[id,GSV_EXCHANGE_CARD2_SEEN_DUKE + influence_to_num(influence_1)] = 1
            game_state[id,GSV_EXCHANGE_SWAP_1_1] = 1
            game_state[id,GSV_EXCHANGE_SWAP_2_2] = 1
        self.game_state = game_state

    def update_gs_action(self, action):
        '''update game state after an action.'''
        self.game_state[-1][GSV_ACTING_PLAYER_0+action.player.player_id] = 1
        if action.target is not None:
            self.game_state[GSV_TARGET_PLAYER_0 + action.target.player_id] = 1
        if type(action) == type(Income): self.game_state[-1][GSV_ACTION_INCOME] = 1
        if type(action) == type(Foreign_Aid): self.game_state[-1][GSV_ACTION_FOREIGN_AID] = 1
        if type(action) == type(Coup): self.game_state[-1][GSV_ACTION_COUP] = 1
        if type(action) == type(Tax): self.game_state[-1][GSV_ACTION_TAX] = 1
        if type(action) == type(Assassinate): self.game_state[-1][GSV_ACTION_ASSASSINATE] = 1
        if type(action) == type(Exchange): self.game_state[-1][GSV_ACTION_EXCHANGE] = 1
        if type(action) == type(Steal): self.game_state[-1][GSV_ACTION_STEAL] = 1


    def get_challenger(self,action):
        '''Check if any player wants to challenge the action, and if so, return the challenger (else return None).'''

        for player in self.players[:-1]:
            if player.declare_challenge(action,self.game_state):
                self.game_state[-1][GSV_CHALLENGE_BY_PLAYER_0 + player.player_id] = 1
                return player

        return None
             
    
    def get_block(self, action):
        '''Check if any player wants to block, and if so, return a Block object (else return None).'''
        if action.counter_characters:
            for player in self.players[:-1]:
                block = player.declare_block(action, self.game_state)
                if block is not None: 
                    if block.character == "Duke":
                        self.game_state[-1][GSV_BLOCK_FOREIGN_AID] = 1
                        self.game_state[-1][GSV_TARGET_PLAYER_0 + player.player_id]
                    elif block.character == "Captain": self.game_state[-1][GSV_BLOCK_STEALING_CAPTAIN] = 1
                    elif block.character == "Ambassador": self.game_state[-1][GSV_BLOCK_STEALING_AMBASSADOR] =1
                    elif block.character == "Contessa": self.game_state[-1][GSV_BLOCK_ASSASSINATION] = 1

                    return block
        return None
    
    def get_block_challenger(self, block):
        '''Check if any player wants to challenge the block, and if so, return the challenger (else return None).'''
        
        block_idx = self.get_player_idx(block.player)
        block_players = self.players[block_idx+1:] + self.players[:block_idx]

        for player in block_players:
            if player.declare_challenge_to_block(block, self.game_state):
                self.game_state[-1][GSV_CHALLENGE_BLOCK_BY_PLAYER_0 + player.player_id] = 1
                return player
        return None
    
    def update_game_state(self, influence):
        ... # TODO
    
        
    def shuffle_deck(self):
        random.shuffle(self.deck)


        
    def do_challenge(self, challenger: Player, move: Move):
        '''Do challenge and return `True` if challenge was succesful (i.e. the player was not allowed to play the action).'''
        player = move.player
        card = move.character
        if move.is_valid():
            influence = challenger.lose_influence()

            # player gets a new card
            self.deck.append(card)
            self.shuffle_deck()
            new_card = self.deck.pop(0)
            if move.player.influence[0] == card and move.player.influence_alive[0]:
                move.player.influence[0] = card
            elif move.player.influence[1] == card and move.player.influence_alive[1]:
                move.player.influence[1] = card
            else:
                ValueError("player does not have card that is object of challenge despite winning challenge")

            return True
        else: 
            influence_lost = player.lose_influence()
            #TODO add GSV stuff to this
            False
            
    def get_player_idx(self, player):
        '''Return `player`'s index in `self.players`.'''
        for idx, cur_player in enumerate(self.players):
            if player == cur_player: return idx
        raise ValueError("`player` is not in `self.players`")
    
    # TODO: keep track of this without looping through each player
    def game_over(self):
        '''
        Return `True` if game is over. 
         If `True`, notify each player and return rewards.
         '''
        num_alive = 0
        for player in self.players:
            num_alive += player.alive
        if num_alive > 1: return False
        if num_alive == 0: raise ValueError('All players are dead') # TODO: make error specific
        return True


    def loop(self):
        '''
        This is the game loop.
        Each iteration is the following pair: (get action, action-challenge, block, block-challenge, do_action).
        '''
        # TODO: game over
        turn = 0
        round = 0
        while not self.game_over():
            turn += 1
            round = turn//5
            print(f"round {round} : turn {turn}")
            if turn % 5 == 0:
                self.game_state  = th.concatenate([self.game_state, th.zeros([1,GAME_STATE_VEC_LEN])],0)
                self.game_state[-1,GSV_ROUND_FF1S] = np.sin(round*(1/OMEGA_1)*2*np.pi)
                self.game_state[-1,GSV_ROUND_FF1C] = np.cos(round*(1/OMEGA_1)*2*np.pi)
                self.game_state[-1,GSV_ROUND_FF2S] = np.sin(round*(1/OMEGA_2)*2*np.pi)
                self.game_state[-1,GSV_ROUND_FF2C] = np.cos(round*(1/OMEGA_2)*2*np.pi)
                self.game_state[-1,GSV_ROUND_FF3S] = np.sin(round*(1/OMEGA_3)*2*np.pi)
                self.game_state[-1,GSV_ROUND_FF3C] = np.cos(round*(1/OMEGA_3)*2*np.pi)
            # Choose the player to play an action. This player will be pushed to the end of `self.players`
            player = self.players[0]
            self.players = self.players[1:] + [player]
            if not player.alive:
                continue
            action = player.declare_action(self.players, self.game_state)
            
            # challenge
            challenger = self.get_challenger(action)
            if (challenger is not None) and self.do_challenge(challenger, action):
                self.game_state[-1][GSV_CHALLENGE_SUCCESS] = 1
                continue
                    
            action.incur_costs()
            
            # block
            block = self.get_block(action)
            if block is not None:
                
                # block-challenge
                block_challenger = self.get_block_challenger(block)
                if (block_challenger is not None) and self.do_challenge(block_challenger, block): 
                    self.game_state[-1][GSV_CHALLENGE_BLOCK_SUCCESS] = 1
                    continue

            action.do_action(self.deck)

        # TODO: return rewards to each player
