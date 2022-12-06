from game import Game
from player import Player
from agent import Agent, RandomAgent
from utils import *
from move import *
import torch as th
import torch.nn as nn
import torch.functional as F

def test_1():
    ITERATIONS = 200
    for j in range(ITERATIONS):
        player_list = []
        deck = DECK.copy()
        random.shuffle(deck)
        for i in range(5):
            player_list.append(Player([deck.pop(0),deck.pop(1)], i,RandomAgent()))
        game = Game(player_list,deck)
        game.loop()
        print(f"Game {j} success")

def test_2():
    agent = Agent()
    random_tensor = th.rand([10000,GAME_STATE_VEC_LEN])
    output = agent(random_tensor,th.ones(ACTION_VEC_LEN),target_mask = th.ones(ACTION_VEC_LEN))
    assert((output[0]).shape[0] == ACTION_VEC_LEN)
if __name__ == "__main__":
    test_2()
    test_1()