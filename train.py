from utils import *
from game import Game
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from player import Player

class Actor:
    ...

class Critic:
    ...

def run_episode(players):
    game = Game(players, DECK.copy())
    
    # TODO: fetch action_probs and game_states from a file instead (or return them?)
    action_probs = 0 # [player, turn, phase, actions]  
    game_states = 0 # TODO
    
    win_id = game.loop()

    rewards = th.zeros(5)
    rewards[win_id] = 1

    # TODO: get values
    values = 0

    return action_probs, values, rewards

def get_actor_loss(action_probs, values, returns):
    advantage = returns - values # TODO
    action_log_probs = th.math.log(action_probs)
    loss = -th.sum(action_log_probs * advantage)

    return loss

def batch_for_critic(game_state,rewards, batch_size):
    '''take a list of game_states and rewards and returns a list batched tensors. Also
    returns a list of how many turns each of the games in each respective batch went for
    (this is necessary because of padding and stuff).'''
    pass

def get_critic_loss(critic, game_states, win_ids):
    # TODO: add batching:
    game_length = game_states.shape[0]
    loss_fun = nn.CrossEntropyLoss()
    total_loss = th.zeros(1)
    # Some phases / turns are very similar to past phases / turns.
    for turn in range(game_length):
        for phase in range(5):
            masked_game_states = temporal_masking(game_states,turn,phase)
            predicted_winners = critic(masked_game_states)
            total_loss += loss_fun(predicted_winners,win_ids)

    return total_loss



def train_step(critic, players, critic_optimizer, actor_optimizers):
    '''Runs a single train step for all players.'''
    action_probs, values, rewards, game_states = run_episode(players)

    # update weights for critic
    critic_optimizer.zero_grad()
    critic_loss = get_critic_loss(...) # TODO
    critic_loss.backward()
    critic_optimizer.step()

    # update weights for each actor
    for player, actor_optimizer in zip(players, actor_optimizers):
        actor = player.actor
        actor_optimizer.zero_grad()
        actor_loss = get_actor_loss(...) # TODO
        actor_loss.backward()
        actor_optimizer.step()

    return critic_loss
    
def train_loop(num_iters):

    # TODO: initialize
    players = [Player(i, Actor()) for i in range(5)]
    critic_optimizer = 0 # TODO
    actor_optimizers = [] # TODO
    episode_rewards = []
    
    for i in range(num_iters):

        # TODO return losses
        episode_reward = train_step(players, critic_optimizer, actor_optimizers)
        episode_rewards.append(episode_reward)
        # print?
        
        # other break conditions