import torch as th



# Action VECTOR is the output tensor of the agent
# not a dict because of autocomplete
AV_NOOP = 0
#Actions
AV_INCOME = 1
AV_FOREIGN_AID = 2
AV_COUP = 3
AV_TAX = 4
AV_ASSASSINATE = 5
AV_EXCHANGE = 6
AV_STEAL = 7
#Targeted players
AV_TARGET_PLAYER_1 = 8
AV_TARGET_PLAYER_2 = 9
AV_TARGET_PLAYER_3 = 10
AV_TARGET_PLAYER_4 = 11
#blocks (includes NoOP)
AV_BLOCK_FOREIGN_AID = 12
AV_BLOCK_STEALING_AMBS = 13
AV_BLOCK_STEALING_CPT = 14
AV_BLOCK_ASSASSINATION = 15
#Challenge or Challenge Block
AV_CHALLENGE = 16
AV_CHALLENGE_BLOCK = 17
# Which cards to keep if attacked
AV_KEEP_CARD_0 = 18
AV_KEEP_CARD_1 = 19
# The distribution of what cards to replace card 0 with
# for exchange
AV_EXCHNG_CARD_0_DUKE = 20
AV_EXCHNG_CARD_0_ASSASSIN = 21
AV_EXCHNG_CARD_0_AMBASSADOR = 22
AV_EXCHNG_CARD_0_CAPTAIN = 23
AV_EXCHNG_CARD_0_CONTESSA = 24
# The distribution of what cards to replace card 1 with
# for exchange
AV_EXCHNG_CARD_1_DUKE = 25
AV_EXCHNG_CARD_1_ASSASSIN = 26
AV_EXCHNG_CARD_1_AMBASSADOR = 27
AV_EXCHNG_CARD_1_CAPTAIN = 28
AV_EXCHNG_CARD_1_CONTESSA = 29

ACTION_VEC_LEN = 30

# Game_State vector is the input to the agent

# which players are acting
GSV_ACTING_PLAYER_0 = 0
GSV_ACTING_PLAYER_1 = 1
GSV_ACTING_PLAYER_2 = 2
GSV_ACTING_PLAYER_3 = 3
GSV_ACTING_PLAYER_4 = 4
# What action did they take
GSV_ACTION_INCOME = 5
GSV_ACTION_FOREIGN_AID = 6
GSV_ACTION_COUP = 7
GSV_ACTION_TAX = 8
GSV_ACTION_ASSASSINATE = 9
GSV_ACTION_EXCHANGE = 10
GSV_ACTION_STEAL = 11
# How much money did the earn / lose 
GSV_COST_PROFIT = 12
# Which player is targeted
GSV_TARGET_PLAYER_0 = 13
GSV_TARGET_PLAYER_1 = 14
GSV_TARGET_PLAYER_2 = 15
GSV_TARGET_PLAYER_3 = 16
GSV_TARGET_PLAYER_4 = 17
# what influence was revealed when an action succeeds
# (IE what card died after being assassinated / couped)
GSV_CARD_REVEALED_ACTION_DUKE = 18
GSV_CARD_REVEALED_ACTION_ASSASSIN = 19
GSV_CARD_REVEALED_ACTION_AMBASSADOR = 20
GSV_CARD_REVEALED_ACTION_CAPTAIN = 21
GSV_CARD_REVEALED_ACTION_CONTESSA = 22
# Who challenged
GSV_CHALLENGE_BY_PLAYER_0 = 23
GSV_CHALLENGE_BY_PLAYER_1 = 24
GSV_CHALLENGE_BY_PLAYER_2 = 25
GSV_CHALLENGE_BY_PLAYER_3 = 26
GSV_CHALLENGE_BY_PLAYER_4 = 27
# Did the challenge succeed
GSV_CHALLENGE_SUCCESS = 28
# What card was revealed as a result of the challenge
GSV_CARD_REVEALED_CHALLENGE_DUKE = 29
GSV_CARD_REVEALED_CHALLENGE_ASSASSIN = 30
GSV_CARD_REVEALED_CHALLENGE_AMBASSADOR = 31
GSV_CARD_REVEALED_CHALLENGE_CAPTAIN = 32
GSV_CARD_REVEALED_CHALLENGE_CONTESSA = 33
# The types of Blocks
GSV_BLOCK_FOREIGN_AID = 34
GSV_BLOCK_STEALING_AMBASSADOR = 35
GSV_BLOCK_STEALING_CAPTAIN = 36
GSV_BLOCK_ASSASSINATION = 37
# Who challenged a block
GSV_CHALLENGE_BLOCK_BY_PLAYER_0 = 38
GSV_CHALLENGE_BLOCK_BY_PLAYER_1 = 39
GSV_CHALLENGE_BLOCK_BY_PLAYER_2 = 40
GSV_CHALLENGE_BLOCK_BY_PLAYER_3 = 41
GSV_CHALLENGE_BLOCK_BY_PLAYER_4 = 42
# Did the challenge block succeed
GSV_CHALLENGE_BLOCK_SUCCESS = 43
# What card was revealed by the challenge block
GSV_CARD_REVEALED_CHALLENGE_BLOCK_DUKE = 44 
GSV_CARD_REVEALED_CHALLENGE_BLOCK_ASSASSIN = 45
GSV_CARD_REVEALED_CHALLENGE_BLOCK_AMBASSADOR = 46 
GSV_CARD_REVEALED_CHALLENGE_BLOCK_CAPTAIN = 47
GSV_CARD_REVEALED_CHALLENGE_BLOCK_CONTESSA = 48
# TODO: REFACTOR THIS.
# represents swapping current influence with drawn influence
GSV_EXCHANGE_SWAP_1_1 = 49
GSV_EXCHANGE_SWAP_1_2 = 50
GSV_EXCHANGE_SWAP_2_1 = 51
GSV_EXCHANGE_SWAP_2_2 = 52
# what cards were drawn in influence
GSV_EXCHANGE_CARD1_SEEN_DUKE = 53
GSV_EXCHANGE_CARD1_SEEN_ASSASSIN = 54
GSV_EXCHANGE_CARD1_SEEN_AMBASSADOR = 55
GSV_EXCHANGE_CARD1_SEEN_CAPTAIN = 56
GSV_EXCHANGE_CARD1_SEEN_CONTESSA = 57
GSV_EXCHANGE_CARD2_SEEN_DUKE = 58
GSV_EXCHANGE_CARD2_SEEN_ASSASSIN = 59
GSV_EXCHANGE_CARD2_SEEN_AMBASSADOR = 60
GSV_EXCHANGE_CARD2_SEEN_CAPTAIN = 61
GSV_EXCHANGE_CARD2_SEEN_CONTESSA = 62
#These "card obtained" are for situations where the
# player who played the action / block won their challenge
# and had to swap for a new card. This allows them to know what card they currently have.
GSV_CARD_OBTAINED_CHALLENGE_DUKE = 63
GSV_CARD_OBTAINED_CHALLENGE_ASSASSIN = 64
GSV_CARD_OBTAINED_CHALLENGE_AMBASSADOR = 65
GSV_CARD_OBTAINED_CHALLENGE_CAPTAIN = 66
GSV_CARD_OBTAINED_CHALLENGE_CONTESSA = 67
GSV_CARD_OBTAINED_CHALLENGE_BLOCK_DUKE = 68
GSV_CARD_OBTAINED_CHALLENGE_BLOCK_ASSASSIN = 69
GSV_CARD_OBTAINED_CHALLENGE_BLOCK_AMBASSADOR = 70
GSV_CARD_OBTAINED_CHALLENGE_BLOCK_CAPTAIN = 71
GSV_CARD_OBTAINED_CHALLENGE_BLOCK_CONTESSA = 72

# Fourier Features for different rounds (s means sine - c means cosine)
GSV_ROUND_FF1S = 73
GSV_ROUND_FF1C = 74
GSV_ROUND_FF2S = 75
GSV_ROUND_FF2C = 76
GSV_ROUND_FF3S = 77
GSV_ROUND_FF3C = 78
GSV_ROUND_FF4S = 79
GSV_ROUND_FF4C = 80
# What is the current phase (for the most recent PHASE)
GSV_PHASE_ACTION = 81
GSV_PHASE_CHALLENGE = 82
GSV_PHASE_BLOCK = 83
GSV_PHASE_CHALLENGE_BLOCK = 84
# Is this round the active round?
GSV_ACTIVE_ROUND = 85
#this part if for the VALUE NETWORK ONLY,
# in order to evaluate wether an action is good or not.
GSV_EXCHANGE_DIST_0_DUKE = 86
GSV_EXCHANGE_DIST_0_ASSASSIN = 87
GSV_EXCHANGE_DIST_0_AMBASSADOR = 88
GSV_EXCHANGE_DIST_0_CAPTAIN = 89
GSV_EXCHANGE_DIST_0_CONTESSA = 90
GSV_EXCHANGE_DIST_1_DUKE = 91
GSV_EXCHANGE_DIST_1_ASSASSIN = 92
GSV_EXCHANGE_DIST_1_AMBASSADOR = 93
GSV_EXCHANGE_DIST_1_CAPTAIN = 94
GSV_EXCHANGE_DIST_1_CONTESSA = 95
GAME_STATE_VEC_LEN = 96

#not my turn mask
NOT_MY_TURN_MASK = th.zeros(GAME_STATE_VEC_LEN)
#player can always see:
#Who acted
NOT_MY_TURN_MASK[GSV_ACTING_PLAYER_0:GSV_ACTING_PLAYER_4 + 1] = 1
#Who was targeted
NOT_MY_TURN_MASK[GSV_TARGET_PLAYER_0:GSV_TARGET_PLAYER_4 + 1] = 1
#Who challenged
NOT_MY_TURN_MASK[GSV_CHALLENGE_BY_PLAYER_0:GSV_CHALLENGE_BY_PLAYER_4+1] = 1
#Who challenged a block
NOT_MY_TURN_MASK[GSV_CHALLENGE_BLOCK_BY_PLAYER_0:GSV_CHALLENGE_BLOCK_BY_PLAYER_4 + 1] = 1
#The nature of a block (if it happened)
NOT_MY_TURN_MASK[GSV_BLOCK_FOREIGN_AID:GSV_BLOCK_ASSASSINATION + 1] = 1
#Wether the challenge succeeded
NOT_MY_TURN_MASK[GSV_CHALLENGE_SUCCESS] = 1
#Wether or not the challenge block succeeded
NOT_MY_TURN_MASK[GSV_CHALLENGE_BLOCK_SUCCESS] = 1
#What action was taken
NOT_MY_TURN_MASK[GSV_ACTION_INCOME: GSV_ACTION_STEAL + 1] = 1
#The associated cost or profit
NOT_MY_TURN_MASK[GSV_COST_PROFIT] = 1
#The fourier features of the round
NOT_MY_TURN_MASK[GSV_ROUND_FF1S: GSV_ROUND_FF3C + 1] = 1
#What phase of the turn it is / wether this round is the "active" round or not.
NOT_MY_TURN_MASK[GSV_PHASE_ACTION:GSV_ACTIVE_ROUND + 1] = 1
#What swaps the opponent made in an exchange.
NOT_MY_TURN_MASK[GSV_EXCHANGE_SWAP_1_1:GSV_EXCHANGE_SWAP_2_2 + 1] = 1
#What card was revealed due to an action:
NOT_MY_TURN_MASK[GSV_CARD_REVEALED_ACTION_DUKE:GSV_CARD_REVEALED_ACTION_CONTESSA + 1] = 1
#What card was revealed due to a challenge:
NOT_MY_TURN_MASK[GSV_CARD_REVEALED_CHALLENGE_DUKE:GSV_CARD_REVEALED_CHALLENGE_CONTESSA + 1] = 1
#What card was revealed due to a block:
NOT_MY_TURN_MASK[GSV_CARD_REVEALED_CHALLENGE_BLOCK_DUKE:GSV_CARD_REVEALED_CHALLENGE_BLOCK_CONTESSA + 1] = 1


#what a player would know if it was their action previously:
WAS_MY_ACTION_MASK = th.zeros(GAME_STATE_VEC_LEN)
#A player would know any cards they saw through exchange:
WAS_MY_ACTION_MASK[GSV_EXCHANGE_CARD1_SEEN_DUKE: GSV_EXCHANGE_CARD2_SEEN_CONTESSA + 1] = 1
#What card they swapped for if they won a challenge:
WAS_MY_ACTION_MASK[GSV_CARD_OBTAINED_CHALLENGE_DUKE: GSV_CARD_OBTAINED_CHALLENGE_CONTESSA + 1] = 1

#what a player would know if it was their block previously:
WAS_MY_BLOCK_MASK = th.zeros(GAME_STATE_VEC_LEN)
WAS_MY_ACTION_MASK[GSV_CARD_OBTAINED_CHALLENGE_BLOCK_DUKE : GSV_CARD_OBTAINED_CHALLENGE_BLOCK_CONTESSA + 1] = 1

PLAYER_ORDERING_PERMUTATIONS = [[0,1,2,3,4] , [1,0,2,3,4] , [2,0,1,3,4], [3,0,1,2,4], [4,0,1,2,3]]

#FREQUENCIES FOR FOURIER FEATURES
OMEGA_1 = 2
OMEGA_2 = 4
OMEGA_3 = 16
OMEGA_4 = 32
DECK = ['Duke', 'Duke', 'Duke', 'Assassin' , 'Assassin' , 'Assassin',  'Ambassador','Ambassador','Ambassador', 'Captain', 'Captain', 'Captain', 'Contessa' , 'Contessa', 'Contessa']

def influence_to_num(influence : str):
    '''convert an influence to a coressponding number, starting at DUKE=0'''
    if influence == 'Duke': return 0
    if influence == 'Assassin': return 1
    if influence == 'Ambassador': return 2
    if influence == 'Captain': return 3
    if influence == 'Contessa': return 4

def num_to_influnece(num : int):
    '''converta  number to a corresponding influence, starting at 0=DUKE'''
    if num == 0: return 'Duke'
    if num == 1: return 'Assassin'
    if num == 2: return'Ambassador'
    if num == 3: return 'Captain'
    if num == 4: return 'Contessa'
    ValueError("number being converted is not 0-4")


def single_sample(distribution : th.Tensor):
    '''Single Sample looks through the whole distribution and finds the one positive value
    That positive value is treated as the probabilty in itself, and is sampled from. '''
    idx = distribution > 0
    assert(sum(idx) == 1)
    return th.rand(1)[0] < distribution[idx]

# Start writing code here...