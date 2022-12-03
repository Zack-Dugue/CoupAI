from game import Game
from player import Player
from agent import RandomAgent
from utils import *
from move import *
import torch as th
import torch.nn as nn
import torch.functional as F

def main():
    player_list = []
    deck = DECK.copy()
    random.shuffle(deck)
    for i in range(5):
        player_list.append(Player((deck.pop(0),deck.pop(1)), i,RandomAgent))
    game = Game(player_list,deck)
    game.loop()
if __name__ == "__main__":
    main()