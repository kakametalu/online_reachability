# This module contains classes to extend the MDP to multi-dimensional grids

import numpy as np
from mdp.mdp_base import MDP
from sklearn.utils.extmath import cartesian
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import product

# TODO (AKA): Update all docstrings!

def state_to_idx(states, num_nodes):
    """Takes states as grid indices and returns vector indices."""
    
    dims = len(num_nodes)
    indices = np.ravel_multi_index(list(states.T), num_nodes)
    return indices

def rectangles_to_states(rectangles):
    states = []
    for rect in rectangles:
        rect_axes = [np.arange(rect_d[0], rect_d[1]) for rect_d in rect]
        states += list(cartesian(rect_axes)) 
    states =  np.array(states)
    return states

class GridWorld(MDP):
    """A class for a simple d-dimensional grid world. 

    A discrete d-dimensional grid where each state (s) is a grid
    point. A state is an array of indices s=[dim1_idx, dim2_idx,...dimd_idx].

    This class also takes in a transition model.
    Attributes:
        _num_nodes (uint): Number of nodes per dimension.
        _all_states (2d np array): All states.
            Size is product(num_nodes) by len(num_nodes). 
        _all_obstacles (2D np array): All obstacle states.
            Size is number of obstacles by len(num_nodes).

    Args:
        num_nodes (list of uints): Number of nodes in each dimension.
        p_trans(3d np array): State transition probabilities in a tensor.
            Tensor dimension is num_actions by num_states by num_states.
            Usage: _p_trans[action, state, next_state].
        all_states (2d np array): All states.
            Size is product(num_nodes) by len(num_nodes).
        all_actions (2d np array): All actions.
        gamma(float): Discount factor for MDP.
    """

    def __init__(self, num_nodes, p_trans, all_states=None,
                 all_actions=None, gamma=None):
        """Initialize GridWorld."""
        
        # Creating state and actions
        dims = len(num_nodes)
        num_actions, num_states, _ = p_trans.shape
        self._dims = dims
        self._num_nodes = np.array(num_nodes)
        if all_states is None:
            state_axes = [np.arange(N_d) for N_d in num_nodes]
            all_states = cartesian(state_axes)
        self._all_states = all_states
        self._all_actions = all_actions
        super().__init__(num_states, num_actions, gamma=gamma, p_trans=p_trans)
        
    def _state_to_idx(self, states):
        """Takes states and returns indices."""
        return state_to_idx(states, self._num_nodes)

    def support(self, state, action_idx):
        """Takes state and action and returns (next states, probs).

        Args:
            state (1D np array): State = [dim1_idx,... dimd_idx].
            action_idx (uint): Action index.
                Should be in the range(num_actions)

        Returns:
            support (list of tuples): Possible next states.
            probs (1D np array): Transition probabilities.
        """

        assert ((0 <= state).all() and
                (state < np.array(self._num_nodes)).all()),\
                "At least one state element is out of range."

        assert (0 <= action_idx < self._num_actions),\
            "Action index is out of range."
        
        state_idx = self._state_to_idx(state)
        support_idxs, probs, _ = super().__getitem__((state_idx, action_idx))
        
        support = self._all_states[list(support_idxs)]

        return support, probs
    
    def _step(self, state, action_idx, deterministic=False):
        """Returns a next state.
        """
        support, probs = self.support(state, action_idx)
        cum_probs = np.cumsum(probs)
        idx = 0
        rand_number = np.random.rand()
        while rand_number > cum_probs[idx]:
            idx += 1        
        next_state = support[idx]
        return next_state

    def simulate(self, start_state, horizon, policy, visualize_on=False,
                 deterministic=False):
        """Simulate trajectory.

        Args:
            start_state(np array): Starting state
            policy(np array): Action for each state.
                Size is size of all states by 1.
            horizon(uint): Length of simulation.
            visualize_on (bool): View the simulation.
            goal(np array): Goal state.
            deterministic (bool): Desired action is used in simulation.

        Returns:
            state_traj (2d np array): The state trajectory.
        """

        state = start_state
        state_traj = [state]

        run_sim = True 
      
        # Run simulation.
        state_idx = self._state_to_idx(state)
        #print(state_idx)
        #print(policy[state_idx])
        for k in range(horizon):
            state = self._step(state, policy[state_idx])
            state_traj.append(state)
            state_idx = self._state_to_idx(state)
    
        if visualize_on:
            self._visualize(state_traj)
        
        return np.array(state_traj) 

    def _visualize(self, state_traj, dims=[0,1]):
        """Visualize the state trajetory in two dimensions."""
        pass


    @property 
    def gamma(self):
        """Return current discount factor."""
        return self._gamma

    @property 
    def dims(self):
        """Return dimension of the state space."""
        return self._dims

    @property 
    def num_nodes(self):
        """Return dimension of the state space."""
        return self._num_nodes

class ReachAvoid(GridWorld):
    """Simple Grid world with a target goal state. 

    This class includes a reward function designed to push the state towards 
    a goal. In particular the trajectory reward function is an infinite horizon sum of discounted rewards, and the immediate reward is just an indicator function on the goal state.

    Attributes:
        _gamma (float): Discount factor in [0, 1)
        _avoid (2D np array): States to be avoided.
        _avoid_set(set of tuples): States to be avoided as a set of tuples.
        _reach (2D np array): States to be reached.
        _reach_set(set of tuples): States to be reached as a set of tuples.

    Args:
        num_nodes (list of uints): Number of nodes in each dimension.
        p_trans(3d np array): State transition probabilities.
        avoid (2D np array): States to be avoided.
        reach (2D np array): States to be reached.
    """

    def __init__(self, num_nodes, p_trans, reach=np.array([]),
                 avoid=np.array([]), gamma=.95, all_states=None,
                 all_actions=None):
        """Initialize MDP Object."""
        
        self._reach = reach
        self._avoid = avoid
        if list(reach) ==[]:
            self._reach_set = reach
        else:
            self._reach_set = set(state_to_idx(reach, num_nodes))   
        if list(avoid) ==[]:
            self._avoid_set = reach
        else:
            self._avoid_set = set(state_to_idx(avoid, num_nodes))        
        
        super().__init__(num_nodes, p_trans, all_states,
                         all_actions, gamma=gamma)
        super().add_abs(self._reach_set) # Add reach set to absorbing set.
        super().add_abs(self._avoid_set) # Add avoid set to absorbing set.
        reward = np.zeros([self._num_states, self._num_actions])
        
        # If no reach set specified design reward for avoid problem.
        if list(reach)==[]:
            reward[list(self._avoid_set),:] = -1.0
        else:
            reward[list(self._reach_set),:] = 1.0 
        self._reward = reward

    def simulate(self, start_state, horizon, policy=None, visualize_on=False,
                 deterministic=False):
        """Simulate trajectory.

        Args:
            start_state(np array): Starting state
            policy(np array): Action for each state.
                Size is size of all states by 1.
            horizon(uint): Length of simulation.
            visualize_on (bool): View the simulation.
            goal(np array): Goal state.
            deterministic (bool): Desired action is used in simulation.

        Returns:
            state_traj (2d np array): The state trajectory.
        """
        
        if policy is None:
            policy = self._pi_opt

        state_traj = super().simulate(start_state, horizon, policy,
                         deterministic=deterministic, visualize_on=False)
        time_out = True
        for state in state_traj:
            if tuple(state) in self._avoid_set:
                print("Failure. Entered avoid set.")
                time_out = False
                break
            if tuple(state) in self._reach_set:
                print("Success! Arrived at reach set.")
                time_out = False
                break
        
        if time_out:
            print("Failure. Simulation timed out.")

        if visualize_on:
            self._visualize_traj(state_traj)
        
        return state_traj 

    def _visualize_traj(self, state_traj, dims=[0,1]):
        """Visualize the state trajetory in two dimensions."""
        
        assert(len(dims)==2), "Visualization requires two dimensions."
        d0, d1 = dims
        temp = np.prod(self._num_nodes)**(1.0/len(self._num_nodes))
        ms = ((100.0 + 2)/(temp + 2)) * 2 # Marker size
        plt.figure(figsize=(8, 8))
        plt.axis([-0.5, self._num_nodes[d0] - 0.5, -0.5,
                  self._num_nodes[d1] - 0.5])
        plt.ion()

        # Prepare legend and title
        plt.plot([],[],'go', markersize= 3, label='Reach')
        plt.plot([],[],'ko', markersize= 3, label='Free space')
        plt.plot([],[],'ro', markersize= 3, label='Avoid')
        plt.plot([],[],'bo', markersize= 3, label='State trajectory')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
        plt.title("Simulated Path", fontsize=20)

        # plot grid
        plt.plot(self._all_states[:, d0], self._all_states[:, d1],
                 'ko', markersize=ms)
            
        # plot avoid
        if list(self._avoid) != []:
            plt.plot(self._avoid[:, 0], self._avoid[:,1],
                     'ro', markersize=ms*1.1)

        # plot reach
        if list(self._reach) != []:
            plt.plot(self._reach[:, 0], self._reach[:,1],
                     'go', markersize=ms*1.1)
        # plot traj
        old_state = state_traj[0]
        for idx, state in enumerate(state_traj):
            scale = 2 if idx == 0 else 1
            plt.plot([old_state[d0], state[d0]],
                     [old_state[d1], state[d1]],'bo-', markersize=ms * scale)  
            old_state = state 
            plt.pause(0.1)
        
        plt.savefig('simulated_path.png')

        plt.pause(3) # Plot will last three seconds
        return

    def visualize_policy(self, policy=None):
        """Visualize the policy.
        
        Args:
            policy(1D np array): Policy to be visualized.
                Size is number of states.
        """

        assert(self._dims==2),\
            "Can only visualize policies for 2D grids."

        plt.figure(figsize=(8, 8))
        plt.axis([-0.5, self._num_nodes[0] - 0.5, -0.5,
                  self._num_nodes[1] - 0.5])
        plt.ion()
        
        # Symbols for each action
        act_symbol = {0:'o', 3:'$\u2192$', 4:'$\u2191$',
                      1:'$\u2190$', 2:'$\u2193$'}

        state_action = np.concatenate([self._all_states,
                                       policy.reshape([-1,1])], axis=1)
 
        temp = np.prod(self._num_nodes)**(1.0/len(self._num_nodes))
        ms = ((100.0 + 2)/(temp + 2)) * 2 # Marker size    
        # plot 
        for act in range(self._num_actions):
            select = state_action[state_action[:,2]==act]
            plt.plot(select[:, 0], select[:,1],'k', 
                     markersize=ms, marker=act_symbol[act],linestyle='None')

        # plot avoid
        if list(self._avoid) != []:
            plt.plot(self._avoid[:, 0], self._avoid[:,1],
                     'ro', markersize=ms*1.1)

        # plot reach
        if list(self._reach) != []:
            plt.plot(self._reach[:, 0], self._reach[:,1],
                     'go', markersize=ms*1.1)

        plt.title("Optimal Policy", fontsize=20)
        plt.savefig('optimal_policy.png')
        plt.pause(3) # Plot will last three seconds

    def visualize_v_func(self, v_func = None, contours=None):
        """Visualize contour plot of value function.

        Args:
            v_func(1D np array): Value function to be visualized.
                Size is number of states.
        """

        assert(self._dims==2 or self._dims==1),\
            "Can only visualize value functions for 1D and 2D grids."

        plt.figure(figsize=(8, 8))
        if self._dims ==1:
            plt.plot(v_func)
        else:

            x = range(self._num_nodes[0])
            y = range(self._num_nodes[1])
        
            if contours is None:
                plt.contour(x, y, v_func.reshape(self._num_nodes).T)
            else:
                plt.contour(x, y, v_func.reshape(self._num_nodes).T, contours)
        plt.ion()
        plt.savefig('value_function.png')
        plt.pause(1) # Plot will last three seconds

    @property 
    def reach(self):
        """Return reach states."""
        return self._reach

    @property 
    def avoid(self):
        """Return avoid states."""
        return self._avoid

    