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
        self.game_state[-1,GSV_ACTING_PLAYER_0+action.player.player_id] = 1
        if action.target is not None:
            self.game_state[-1,GSV_TARGET_PLAYER_0 + action.target.player_id] = 1
        if type(action) == type(Income): self.game_state[-1,GSV_ACTION_INCOME] = 1
        if type(action) == type(Foreign_Aid): self.game_state[-1,GSV_ACTION_FOREIGN_AID] = 1
        if type(action) == type(Coup): self.game_state[-1,GSV_ACTION_COUP] = 1
        if type(action) == type(Tax): self.game_state[-1,GSV_ACTION_TAX] = 1
        if type(action) == type(Assassinate): self.game_state[-1,GSV_ACTION_ASSASSINATE] = 1
        if type(action) == type(Exchange):
            self.game_state[-1,GSV_ACTION_EXCHANGE] = 1
            self.game_state[-1,GSV_EXCHANGE_DIST_0_DUKE:GSV_EXCHANGE_DIST_1_CONTESSA+1] = action.influence_dist[AV_EXCHNG_CARD_0_DUKE:AV_EXCHNG_CARD_1_CONTESSA+1]
        if type(action) == type(Steal): self.game_state[-1,GSV_ACTION_STEAL] = 1


    def get_challenger(self,action):
        '''Check if any player wants to challenge the action, and if so, return the challenger (else return None).'''
        self.game_state[-1,GSV_PHASE_CHALLENGE] = 1
        for player in self.players[:-1]:
            if player.alive and player.declare_challenge(action,self.game_state):
                self.game_state[-1][GSV_CHALLENGE_BY_PLAYER_0 + player.player_id] = 1
                self.game_state[-1, GSV_PHASE_CHALLENGE] = 0
                return player
        self.game_state[-1,GSV_PHASE_CHALLENGE] = 0

        return None



    def get_block(self, action):
        '''Check if any player wants to block, and if so, return a Block object (else return None).'''
        self.game_state[-1,GSV_PHASE_BLOCK] = 1
        if action.counter_characters:
            for player in self.players[:-1]:
                if not player.alive: continue
                block = player.declare_block(action, self.game_state)
                if block is not None: 
                    if block.character == "Duke":
                        self.game_state[-1][GSV_BLOCK_FOREIGN_AID] = 1
                        self.game_state[-1][GSV_TARGET_PLAYER_0 + player.player_id]
                    elif block.character == "Captain": self.game_state[-1][GSV_BLOCK_STEALING_CAPTAIN] = 1
                    elif block.character == "Ambassador": self.game_state[-1][GSV_BLOCK_STEALING_AMBASSADOR] =1
                    elif block.character == "Contessa": self.game_state[-1][GSV_BLOCK_ASSASSINATION] = 1
                    self.game_state[-1,GSV_PHASE_BLOCK] = 0
                    return block
        self.game_state[-1,GSV_PHASE_BLOCK] = 0
        return None
    
    def get_block_challenger(self, block):
        '''Check if any player wants to challenge the block, and if so, return the challenger (else return None).'''
        
        block_idx = self.get_player_idx(block.player)
        block_players = self.players[block_idx+1:] + self.players[:block_idx]
        self.game_state[-1,GSV_PHASE_CHALLENGE_BLOCK] = 1
        for player in block_players:
            if player.alive and player.declare_challenge_to_block(block, self.game_state):
                self.game_state[-1,GSV_CHALLENGE_BLOCK_BY_PLAYER_0 + player.player_id] = 1
                self.game_state[-1,GSV_PHASE_CHALLENGE_BLOCK] = 0
                return player
        self.game_state[-1,GSV_PHASE_CHALLENGE_BLOCK] = 0
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
            if issubclass(move.__class__,Action):
                self.game_state[-1,GSV_CARD_REVEALED_CHALLENGE_DUKE + influence_to_num(influence)] = 1
            elif issubclass(move.__class__,Block):
                self.game_state[-1,GSV_CARD_REVEALED_CHALLENGE_BLOCK_DUKE + influence_to_num(influence)] = 1


            # player gets a new card
            self.deck.append(card)
            self.shuffle_deck()
            new_card = self.deck.pop(0)
            if issubclass(move.__class__, Action):
                self.game_state[-1, GSV_CARD_OBTAINED_CHALLENGE_DUKE + influence_to_num(new_card)] = 1
            elif issubclass(move.__class__, Block):
                self.game_state[-1, GSV_CARD_OBTAINED_CHALLENGE_BLOCK_DUKE + influence_to_num(influence)] = 1

            if move.player.influence[0] == card and move.player.influence_alive[0]:
                move.player.influence[0] = new_card
            elif move.player.influence[1] == card and move.player.influence_alive[1]:
                move.player.influence[1] = new_card
            else:
                ValueError("player does not have card that is object of challenge despite winning challenge")

            return False
        else: 
            influence_lost = player.lose_influence()
            #TODO add GSV stuff to this
            return    True
            
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
            self.check_game()
            self.game_state[-1,GSV_ACTIVE_ROUND] = 0
            self.game_state  = th.concatenate([self.game_state, th.zeros([1,GAME_STATE_VEC_LEN])],0)
            self.game_state[-1,GSV_ROUND_FF1S] = np.sin(round*(1/OMEGA_1)*2*np.pi)
            self.game_state[-1,GSV_ROUND_FF1C] = np.cos(round*(1/OMEGA_1)*2*np.pi)
            self.game_state[-1,GSV_ROUND_FF2S] = np.sin(round*(1/OMEGA_2)*2*np.pi)
            self.game_state[-1,GSV_ROUND_FF2C] = np.cos(round*(1/OMEGA_2)*2*np.pi)
            self.game_state[-1,GSV_ROUND_FF3S] = np.sin(round*(1/OMEGA_3)*2*np.pi)
            self.game_state[-1,GSV_ROUND_FF3C] = np.cos(round*(1/OMEGA_3)*2*np.pi)
            self.game_state[-1,GSV_ACTIVE_ROUND] = 1
            # Choose the player to play an action. This player will be pushed to the end of `self.players`
            player = self.players[0]
            self.players = self.players[1:] + [player]
            if not player.alive:
                print(f"\t player {player.player_id} is dead")
                continue
            self.game_state[-1,GSV_PHASE_ACTION] = 1
            action = player.declare_action(self.players, self.game_state)
            self.game_state[-1,GSV_PHASE_ACTION] = 0

            self.update_gs_action(action)
            if action.target is not None:
                target_id = action.target.player_id
            else:
                target_id = None
            print(f"\t player {player.player_id} is doing action {action} with target {target_id}")

            # challenge
            challenger = self.get_challenger(action)
            if challenger is not None:
                print(f"\tPlayer {challenger.player_id} is doing challenge")
            else:
                print(f"\tno challenge")

            if (challenger is not None) and self.do_challenge(challenger, action):
                self.game_state[-1,GSV_CHALLENGE_SUCCESS] = 1
                print(f"\tChallenge was a success!")
                continue
            self.check_game()

            action.incur_costs()
            
            # block

            block = self.get_block(action)
            if block is not None:
                print(f"\t player {block.player.player_id} is blocking with {block.character}")
                # block-challenge
                block_challenger = self.get_block_challenger(block)
                if block_challenger is not None:
                    print(f"\t block is challenged by {block_challenger.player_id}")
                if (block_challenger is not None) and self.do_challenge(block_challenger, block):
                    self.game_state[-1,GSV_CHALLENGE_BLOCK_SUCCESS] = 1
                    print(f"\t challenge to block succeeds")
                else:
                    print(f"\t challenge to block failse")
                    continue

            self.check_game()
            action.do_action(self.deck,self.game_state)

        # TODO: return rewards to each player
    def check_game(self):
        # check the number of cards:
        count_dict = {"Duke" : 0 , "Assassin": 0, "Ambassador" : 0, "Captain" : 0, "Contessa" : 0}
        deck_count_dict = {"Duke" : 0 , "Assassin": 0, "Ambassador" : 0, "Captain" : 0, "Contessa" : 0}
        player_count_dict = {"Duke" : 0 , "Assassin": 0, "Ambassador" : 0, "Captain" : 0, "Contessa" : 0}
        for character in self.deck:
            count_dict[character] += 1
            deck_count_dict[character] += 1
        for player in self.players:
            count_dict[player.influence[0]] += 1
            count_dict[player.influence[1]] += 1
            player_count_dict[player.influence[0]] += 1
            player_count_dict[player.influence[1]] += 1

        assert(3 == count_dict["Duke"] == count_dict["Assassin"] ==
             count_dict["Ambassador"] == count_dict["Captain"] == count_dict["Contessa"])