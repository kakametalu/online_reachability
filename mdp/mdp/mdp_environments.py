# Library containing various MDPs derived from the MDP class in mdp_base

import numpy as np
from mdp.mdp_base import MDP

class RandomWalk1D(MDP):
    """Controlled random walk on a line.

    For each action there is a probability of incrementing the state by 
    -1 (left), 0 (stay), or 1 (right). The end states are absorbing.

    Attributes:
        _dir_probs (2D np array): Left, stay, right probs for each action.
            Size is number of actions by 3.

    Args:
        num_states(uint): Number of states
        dir_probs (2D np array): Left, stay, right probs for each action.
            Size is number of actions by 3.
        reward_func (func): Reward function for MDP.
        gamma (float): Discount factor for MDP.
        absorbing (list of uints): Absorbing state indices.
    """

    def __init__(self, num_states, dir_probs, 
    	         reward_func = None, gamma = None, absorbing = None):
        """Initializing model."""
        num_actions = dir_probs.shape[0]
        super().__init__(num_states, num_actions, reward_func, gamma,
                         absorbing)
        self._dir_probs = dir_probs

        # Populate transition probabilities
        for state in range(num_states):                
            next_states = [state - 1, state, state + 1]
            if state == 0 or state == num_states - 1 :
                next_states = [state, state, state]

            for action in range(num_actions):
                super().add_transition(state, action, next_states,
                	                   dir_probs[action,:])
    
    @property
    def dir_probs(self):
        return self._dir_probs
 