import numpy as np
from mdp.mdp_base import MDP
from sklearn.utils.extmath import cartesian
from copy import deepcopy
import matplotlib.pyplot as plt
from itertools import product

def state_to_idx(states, num_nodes):
    """Takes states and returns indices."""
    
    dims = len(num_nodes)
    indices = np.ravel_multi_index(list(states.T), num_nodes)
    return indices


class GridWorld(MDP):
    """A class for a simple d-dimensional grid world. 

    A discrete d-dimensional grid where each state (s) is a grid
    point. A state is a tuple of indices s=(dim1_idx, dim2_idx,...dimd_idx). There are 2d + 1 desired actions: staying in place and moving to adjacent states (not including diagonal neighbors).
    
    actions = { a in (-1, 0, 1)^d| ||a||_0<=1 } e.g. for 2D a grid the actions are {[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]}.
    
    next_state = state + action
    
    Along the boundaries if a transition is not possible the system stays in
    the same state. Absorbing states only transition to themselves.

    Attributes:
        _action_probs (2d Np array): Action probabilites for a desired action.
            Element ij of this matrix correponds to the probability of taking
            action j when action i is desired.
        _obs_set (set of ints): Checks if state is obstacle.
        _num_nodes (uint): Number of nodes per dimension.
        _all_states (2d np array): All states.
            Size is product(num_nodes) by len(num_nodes). 
        _all_obstacles (2D np array): All obstacle states.
            Size is number of obstacles by len(num_nodes). 

    Args:
        num_nodes (list of uints): Number of nodes in each dimension.
        action_probs (2d Np array): Action probabilites for a desired action.
            Element ij of this matrix correponds to the probability of taking
            action j when action i is desired. Matrix is deterministic by 
            default (i.e. identity matrix).
        obstacles (3D np array): Obstacles in the environment.
            Dimension 0 indexes the obstacles. For each obstacle there is 
            a matrix of size number of dimensions by 2 that defines a
            hyper-rectangle. The first column contains the minimum index along each dimension and the second column contains the maximum index.
    """

    def __init__(self, num_nodes, action_probs=None, obstacles=[],
                 reward=None):
        """Initialize GridWorld."""
        
        # Creating state and actions
        dims = len(num_nodes)
        self._dims = dims
        self._num_nodes = np.array(num_nodes)
        state_axes = [np.arange(N_d) for N_d in num_nodes]
        self._all_states = cartesian(state_axes)
        self._all_actions = np.concatenate([np.zeros([1, dims]),np.eye(dims),                                - np.eye(dims)], axis=0)\
                                            .astype(int)

        num_states = np.prod(self._num_nodes)
        num_actions = 2 * dims + 1
        
        # Prepare transition probabilities
        if action_probs is None:
            action_probs = np.eye(num_actions)
        self._action_probs = action_probs
        p_trans = np.zeros([num_actions, num_states, num_states])        
        state_idxs = range(np.prod(self._num_nodes))
        
        for a_idx in range(num_actions): # a_idx : action index
            action = self._all_actions[a_idx]
            next_states = np.minimum(np.maximum(self._all_states + action, 0),
                                     self._num_nodes - 1)
            next_idxs = list(state_to_idx(next_states, self._num_nodes))
            for da_idx in range(num_actions): # da_idx: desired action index
                p_trans[da_idx, state_idxs, next_idxs] += action_probs[da_idx,
                                                                      a_idx]

        # Creating obstacles
        obstacle_states = []
        for obs in obstacles:
            obs_axes = [np.arange(obs_dim[0], obs_dim[1] + 1)
                        for obs_dim in obs]
            obstacle_states += list(cartesian(obs_axes)) 
        self._all_obstacles =  np.array(obstacle_states)  
        self._obs_set = set(state_to_idx(self._all_obstacles, num_nodes)) 
        super().__init__(num_states, num_actions, reward=reward, gamma=0.95,
                         abs_set= deepcopy(self._obs_set), p_trans=p_trans)
        

        # add some stuff for the goal


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
        idx = 0;
        rand_number = np.random.rand()
        
        while rand_number > cum_probs[idx]:
            idx += 1
        
        if deterministic: # Use desired action.
            idx = action_idx
        
        next_state = support[idx]
        return next_state

    def simulate(self, start_state, horizon, policy, visualize_on=False,
                 goal=None, deterministic=False):
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

        # Check if starting at goal or obstacle before running sim.
        if self._state_to_idx(state) in self._obs_set:
            print("Failure. Hit obstacle.")
            run_sim = False

        if tuple(state) == tuple(goal):
            print("Success! Goal reached.")
            run_sim = False

        if run_sim is False:
            if visualize_on:
                self._visualize(state_traj, goal)
            return        

        # Run simulation.
        time_out = True # Sim breaks early if goal/obstacle is reached.
        state_idx = self._state_to_idx(state)
        for k in range(horizon):
            action = policy[state_idx]
            state = self._step(state, action, deterministic)
            state_traj.append(state)
            state_idx = self._state_to_idx(state)

            if state_idx in self._obs_set:
                print("Failure. Hit obstacle.")
                time_out = False
                break
            
            if tuple(state) == tuple(goal):
                print("Success! Goal reached.")
                time_out = False
                break
    
        if time_out:
            print("Failure. Simulation timed out.")

        if visualize_on:
            self._visualize(state_traj, goal)
        
        return state_traj  

    def _visualize(self, state_traj, goal, dims=[0,1]):
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
        plt.plot([],[],'go', markersize= 3, label='goal')
        plt.plot([],[],'bo', markersize= 3, label='free space')
        plt.plot([],[],'ko', markersize= 3, label='obstacles')
        plt.plot([],[],'ro', markersize= 3, label='state trajectory')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
        plt.title("Simulated Path", fontsize=20)

        # plot grid
        plt.plot(self._all_states[:, d0], self._all_states[:, d1],
                 'bo', markersize=ms)
            
        # plot obstacles
        if list(self._all_obstacles) != []:
            plt.plot(self._all_obstacles[:, d0], self._all_obstacles[:,d1],
                     'ko', markersize=ms)
        
        # plot goal
        if goal is not None:
            plt.plot(goal[d0], goal[d1], 'go', markersize=ms * 2)

        # plot traj
        old_state = state_traj[0]
        for idx, state in enumerate(state_traj):
            scale = 2 if idx == 0 else 1
            plt.plot([old_state[d0], state[d0]],
                     [old_state[d1], state[d1]],'ro-', markersize=ms * scale)  
            old_state = state 
            plt.pause(0.1)
        
        plt.savefig('simulated_path.png')

        plt.pause(3) # Plot will last three seconds
        return

class ReachGoal(GridWorld):
    """Grid world with a target goal. 

    This class includes a reward function designed to push the state towards 
    a goal. In particular the trajectory reward function is an infinite horizon sum of discounted rewards, and the immediate reward is just an indicator function on the goal state.

    Attributes:
        _goal (np array): Goal state.
        _gamma (float): Discount factor in [0, 1)

    Args:
        num_nodes (list of uints): Number of nodes in each dimension.
        action_probs (2d Np array): Action probabilites for a desired action.
            Element ij of this matrix correponds to the probability of taking
            action j when action i is desired. Matrix is deterministic by 
            default (i.e. identity matrix).
        obstacles (3D np array): Obstacles in the environment.
            Dimension 0 indexes the obstacles. For each obstacle there is 
            a matrix of size number of dimensions by 2 that defines a
            hyper-rectangle. The first column contains the minimum index along each dimension and the second column contains the maximum index.
        goal (np array): Goal state.
    """

    def __init__(self, num_nodes, action_probs=None,
                 obstacles=None, goal=None, gamma=.95):
        """Initialize MDP Object."""
        if goal is not None:
            assert ((0 <= goal).all() and
                    (goal < np.array(num_nodes)).all()),\
                "At least one goal element is out of range."
        super().__init__(num_nodes, action_probs, obstacles)
        self._goal = goal
        self._gamma = gamma
    
    def _compute_exp_reward(self):
        """Compute expected immediate reward."""

        goal_idx = self._state_to_idx(self._goal)
        self._reward = lambda s,a,s_next: float(s == goal_idx)
        super()._compute_exp_reward()

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
        act_symbol = {0:'o', 1:'$\u2192$', 2:'$\u2191$',
                      3:'$\u2190$', 4:'$\u2193$'}

        state_action = np.concatenate([self._all_states,
                                       policy.reshape([-1,1])], axis=1)
 
        temp = np.prod(self._num_nodes)**(1.0/len(self._num_nodes))
        ms = ((100.0 + 2)/(temp + 2)) * 2 # Marker size    
        # plot 
        for act in range(self._num_actions):
            select = state_action[state_action[:,2]==act]
            plt.plot(select[:, 0], select[:,1],'k', 
                     markersize=ms, marker=act_symbol[act],linestyle='None')

        # plot obstacles
        if list(self._all_obstacles) != []:
            plt.plot(self._all_obstacles[:, 0], self._all_obstacles[:,1],
                     'ko', markersize=ms*1.1)

        plt.title("Optimal Policy", fontsize=20)
        plt.savefig('optimal_policy.png')
        plt.pause(3) # Plot will last three seconds

    def simulate(self, start_state, horizon, policy=None, visualize_on=False,
                 goal=None, deterministic=False):   
        """Simulate trajectory.

        Args:
            start_state(np array): Starting state
            policy(2D np array): Action for each state.
            horizon(uint): Length of simulation.
            visualize_on (bool): View the simulation.
            goal(np array): Goal state.
            deterministic (bool): Desired action is used in simulation.

        Returns:
            state_traj (2d np array): The state trajectory.
        """
       
        # Set goal
        if goal is not None:
            self.goal = goal # Set new goal.

        if policy is None:
            policy = self.pi_opt
        
        super().simulate(start_state, horizon, policy, 
                         visualize_on=visualize_on, goal=self._goal,
                         deterministic=deterministic)

    @property 
    def goal(self):
        """Return current goal state."""
        return self._goal

    @goal.setter
    def goal(self, goal):
        """Set new goal and compute optimal policy and value function."""
        assert ((0 <= goal).all() and
                (goal < np.array(self._num_nodes)).all()),\
            "At least one goal element is out of range."
        
        if self._goal is not None:
            if tuple(goal) == tuple(self._goal): 
                return # No need to recompute value function and opt policy
        
        # Change goals and update set of absorbing states.
        goal_idx = self._state_to_idx(self._goal)
        if goal_idx in self._abs_set:
            self._abs_set.remove(goal_idx)
        self._goal = goal
        self.add_abs(self._state_to_idx(self._goal))
        
        # Compute new value function and policy.
        self._v_opt, self._pi_opt = self.v_pi_opt(force_run=True)
        return

    @property 
    def gamma(self):
        """Return current discount factor."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """Set new gamma and compute optimal policy and value function."""
        eps = 10 ** -6
        self._goal = min(max(eps,gamma), 1 - eps)
        self._v_opt, self. _pi_opt = self.v_pi_opt(force_run=True)
        return

    