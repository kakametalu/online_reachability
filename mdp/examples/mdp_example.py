# This module provides a simple example of how to use the MDP class

import numpy as np
from mdp.tools import MDP
	
class LeftStayRight(MDP):
    """Simple model for moving to adjacent states.

    For each action there is a probability of incrementing the state by 
    -1 (left), 0 (stay), or 1 (right). The end states are absorbing.

    Attributes:
        _a_lsr (2D np array): Left, stay, right probabilities for each action.
            Size is number of actions by 3.

    Args:
        num_states(uint): Number of states
        a_lsr (2D np array): Left, stay, right probabilities for each action.
            Size is number of actions by 3.
        reward_func (func): Reward function for MDP.
        gamma (float): Discount factor for MDP.
    """

    def __init__(self, num_states, a_lsr, reward_func = None, gamma = None):
        """Initializing model."""
        num_actions = a_lsr.shape[0]
        super().__init__(num_states, num_actions, reward_func, gamma)
        self._a_lsr = a_lsr

        # Populate transition probabilities
        for state in range(num_states):                
            next_states = [state - 1, state, state + 1]
            if state == 0 or state == num_states - 1 :
                next_states = [state, state, state]

            for action in range(num_actions):
                super().add_transition(state, action, next_states,
                	                   a_lsr[action,:])
    
    @property
    def a_lsr(self):
        return self._a_lsr
 
def end_indicator(s, a, s_next): 
    """Indicator for transitioning to last state.""" 
    if s_next == nS-1 and s_next != s:
        return 1.0
    return 0.0

if __name__ == "__main__":
    
    # Initialize a LeftStayRight Model
    nS = 10
    a_lsr = np.array([[0.2, 0.3, 0.5],
                      [0.0, 0.8, 0.2],
                      [0.4, 0.5, 0.1]])

    my_mdp = LeftStayRight(nS, a_lsr, end_indicator)

    print(my_mdp._expR)
