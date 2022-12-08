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

def run_episode(players, critic):
    game = Game(players, DECK.copy(), critic=critic)
    
    # TODO: fetch action_probs and game_states from a file instead (or return them?)
    action_probs = 0 # [player, turn, phase, actions]  
    game_states = 0 # TODO
    
    win_id = game.loop()

    rewards = th.zeros(5)
    rewards[win_id] = 1

    # TODO: get values
    values = 0

    return action_probs, values, rewards

def generate_samples(players,num_batches,batch_size):
    '''Generate multiple samples and batch them.
    Returns a list of 'batches' of game_states, values, and rewards'''
    game_state_list = []
    values_list = []
    reward_list = []
    # the contents of this list will be a tensor for each batch stating how long
    # the game was. This will be important to determining the loss when batching later.
    number_of_turns_list = []
    for j in range(num_batches):
        game_state_batch = []
        values_batch = []
        reward_batch = []
        for i in range(batch_size):
            game_state, values, reward = run_episode(players)
            game_state_batch.append(game_state)
            values_batch.append(values)
            reward_batch.append(reward)
        # TODO: The values list and game_state list
        #  aren't the same size, and thus can't be stacked.
        #  Once I have a better idea what form they're in, I can 0 pad them
        #  to make them stack.
        game_state_list.append(th.stack(game_state_batch,dim=0))
        values_list.append(th.stack(values_batch,dim=0))
        reward_list.append(th.stack(reward_batch,dim=0))

        print(f"game{j} finished")


    return game_state_list, values_list, reward_list, number_of_turns_list


def get_actor_loss(action_probs, values, returns):
    advantage = returns - values # TODO
    action_log_probs = th.math.log(action_probs)
    loss = -th.sum(action_log_probs * advantage)

    return loss

#TODO: consider training critic with bootstrap method if this doesn't work.


def get_critic_loss(critic, game_states, win_ids):
    # TODO: add batching:
    game_length = game_states.shape[0]
    loss_fun = nn.CrossEntropyLoss()
    total_loss = th.zeros(1)
    # Some phases / turns are very similar to past phases / turns.
    # might want to not actually calculate for every single phase
    # or every single turn, as recent turns / phases are likely to be highly correlated.
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