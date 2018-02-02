import numpy as np
from mdp.mdp_base import MDP
from mdp.grid_world import state_to_idx

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
    	         reward = None, gamma = None, absorbing = None):
        """Initializing model."""
        num_actions = dir_probs.shape[0]

        self._dir_probs = dir_probs

        # Populate transition probabilities
        p_trans = np.zeros([num_actions, num_states, num_states])
        next_states = np.minimum(np.maximum(np.arange(num_states) + 1, 0),
            num_states - 1)
        prev_states = np.minimum(np.maximum(np.arange(num_states) - 1, 0),
            num_states - 1)
        for i, action in enumerate(dir_probs):
            p_trans[i,range(num_states), list(prev_states)] += action[0]
            p_trans[i,range(num_states), range(num_states)] += action[1]
            p_trans[i,range(num_states), list(next_states)] += action[2]
        super().__init__(num_states, num_actions, reward, gamma,
                         absorbing, p_trans)

    
    @property
    def dir_probs(self):
        return self._dir_probs
 