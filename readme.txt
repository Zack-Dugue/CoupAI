# Critic - Value Network
    
# Actor - Agent

1) Run the agent on the environment to collect training data per episode (action_probs, critic value, rewards).
    - Run the game
    - Save the Game States 
    - Save the Action Vectors (pair them with the game_state at the time the action occured?)

2) Compute expected return at each time step.
    - discounted return (i.e. G_t = w g^(T-t) where w=1 if win, else 0, and T is the final step, and t is the current step)
    - can set g = 1?
3) Compute the loss for the combined Actor-Critic model.
    - Loss for critic is just Cross Entropy loss of predicted probability of winning
      vs wether the player actually won or not.
    - Loss for actor is the  
4) Compute gradients and update network parameters.
- train Critic to match V to G_t
- Loss = Sum over all Actions: ln(Porbability of Action) * Advantage of Action
    - where Advantage is the Value of an action minus the Average Value over all possible actions

Repeat 1-4 until either success criterion or max episodes has been reached.


When are the Actors Evaluating?:
    1. The Acting Player Evaluates first to decide what action take
    2. If the action has a character: All players except for the acting player evaluate on wether or not to challenge.
    3. If the action is foreign_aid all players evaluate wether or not to block, else only the target evaluate
    4. If there is a block all players (except the blocking player) evaluate wether or not to challenge.


[Player, Turn, Phase, Action Length]
phase number of tensors:
    [Player, Turn, Action Dimension]

Action Dimension of Acting Player is: 
    (Actions no Target (3) + Actions with Target(3)  x Targets (5) + Exchange (1) x Possible Card Swaps (10) ) x Prefered Influence (2) = 56

Action Dimension of Challenging player is:
    Challenge (2) x Prefered Influence (2) = 4

Action Dimension of blocking player is:
    if action is not steal:

        block (2) x Prefered Influence (2) = 4
    
    if action is steal:
        block (3) x Prefered Influence (2) = 6 

Action Dimension of Player Challenging Block is: 
    Challenge (2) x Prefered Influence (2) = 4


VALUE NETWORK EVALUATIONS:
Before the Action
After The Action
After the Challenge
After the Block
After the Challenge of the Block

values (from game) = [Player, Turn, Phase, ACTION_VEC_LEN]


How do we seperate these in game_state
One approach is to mask them: 

Make 5 copies of Games_State


FOR THE ACTORS:
Action, challenge, block, challenge of block

Charlotte:
- make sure game state returns the right things
- run tests on game state vec to make sure things aint broke

Zack:
    Critic Loss
    Phase Masks

TODO:
# What the Game Loop Returns
    - game_state
    - history of action probabilities [by phase]
    - who won
    - values (for each action for each player for each turn for each phase)

# What the Critic Trainer has to do:
    # take game states and win_loss labels
    # return cross entropy loss of both of those

# 
