# This module contains differerent transition models for a discrete state and
# action MDP. In general each transition model should return a probability
# transition tensor (p_trans).
import numpy as np
from sklearn.utils.extmath import cartesian
from mdp.grid_world import state_to_idx

def random_walk_model(num_nodes, action_probs=None):
    """ Controlled random walk model over a d-dimensional grid. 

        There are 2d + 1 desired actions: stay in place and move to adjacent states (not including diagonal neighbors).
        
        actions = { a in (-1, 0, 1)^d| ||a||_0<=1 } e.g. for 2D a grid the actions are {[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]}.
        
        next_state = state + action

        Along the boundaries if a transition is not possible the system stays in the same state.

        Args:
            num_nodes (uint): Number of nodes per dimension.
            action_probs (2d Np array): Action probs for a desired action.
                Element ij of this matrix correponds to the probability of taking action j when action i is desired. Matrix is deterministic by default (i.e. identity matrix).
        Return:
            p_trans(3d np array): State transition probabilities in a tensor.
                Tensor dimension is num_actions by num_states by num_states.
                Usage: _p_trans[action, state, next_state].
    """

    dims = len(num_nodes)
    all_actions = np.concatenate([np.zeros([1, dims]), -np.eye(dims), 
                                  np.eye(dims)],axis=0).astype(int)
    num_states = np.prod(num_nodes)
    state_axes = [np.arange(N_d) for N_d in num_nodes]
    all_states = cartesian(state_axes)
    num_actions = 2 * dims + 1

    # Prepare transition probabilities
    if action_probs is None:
        action_probs = np.eye(num_actions)
    else:
        assert ((action_probs >= 0).all() and 
                (np.abs(np.sum(action_probs,axis=1)-1.0)<(10**-6)).all())\
                ,"Probabilities not valid."
        
    p_trans = np.zeros([num_actions, num_states, num_states])        
    state_idxs = range(np.prod(num_nodes))

    for a_idx in range(num_actions): # a_idx : action index
        action = all_actions[a_idx]
        next_states = np.minimum(np.maximum(all_states + action, 0),
                                     np.array(num_nodes) - 1)
        next_idxs = list(state_to_idx(next_states, np.array(num_nodes)))
        for da_idx in range(num_actions): # da_idx: desired action index
            p_trans[da_idx,state_idxs,next_idxs] += action_probs[da_idx,a_idx]

    return p_trans, all_states, all_actions