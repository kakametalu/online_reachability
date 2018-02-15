# This module contains classes that extend the GridWorld based classes in
# grid_world.py to approximations of continous spaces. 

import numpy as np
from mdp.grid_world import GridWorld as GridWorldDiscrete, state_to_idx
from mdp.dynamics import Dynamics
from mdp.signed_distance import dist_hypercube_int
from sklearn.utils.extmath import cartesian
import time
from copy import deepcopy


class GridWorld(GridWorldDiscrete):
    """Extends GridWorldDiscrete class to continuous state spaces.

    Attributes:
        s_lims(2d np array): Range on the states. 
            First (second) column lower (upper) bound for each state dim.
        a_lims(2d np array): Range on the actions. 
            First (second) column lower (upper) bound for each action dim.
        ds(1d np array): Grid spacing for each state dim.
        da(1d np array): Grid spacing for each action dim.
        dt(float): Time step.
    
    Args:
        num_nodes (1D np array): Number of nodes in each state dimension.
        s_lims(2d np array): Range on the states. 
            First (second) column lower (upper) bound for each state dim.
        num_nodes_a (1D np array): Number of nodes in each action dimension.
        a_lims(2d np array): Range on the actions. 
            First (second) column lower (upper) bound for each action dim.
        dynamics(function): Continous dynamics model.
             Function takes in state (x) and action(a), and returns
             the state derivative (x_dot) 
        gamma(float): Discount factor.
    """

    def __init__(self, num_nodes, s_lims, num_nodes_a, a_lims=None, dynamics=None, gamma=None):

        # Preparing grid
        self._a_lims = s_lims
        self._s_lims = s_lims
        s_min = s_lims[0, :]
        s_max = s_lims[1, :]
        a_min = a_lims[0, :]
        a_max = a_lims[1, :]
        ds = (s_max - s_min)/(num_nodes - 1)
        da = (a_max - a_min)/(num_nodes_a - 1) 
        self._ds = ds
        self._da = ds

        self._num_nodes_a = num_nodes_a

        num_states = np.prod(num_nodes)
        num_actions = np.prod(num_nodes_a)
        dims = len(num_nodes)
        deriv = np.zeros([num_actions, num_states, dims])
        p_trans = np.zeros([num_actions, num_states, num_states])


        # All states (actions) as grid indices
        state_axes = [np.arange(N_d) for N_d in num_nodes]
        all_states = cartesian(state_axes) * ds + s_min

        action_axes = [np.arange(N_d) for N_d in num_nodes_a]
        all_actions = cartesian(action_axes) * da + a_min

        # Construct interpolation weights (transition probabilities)
        dyn = Dynamics(dynamics, num_nodes.size) # dynamics model


        # Hypercube defining interpolation region
        interp_axes = [np.array([0,1]) for d in range(dims)]
        interp_region = cartesian(interp_axes).astype(int)

        for act_idx, action in enumerate(all_actions):
            deriv[act_idx,:,:] = dyn.deriv(all_states, action)

        # State moves at most one grid cell in any dimension over one time
        # step.

        dt = (1.0 / np.amax(np.abs(deriv.reshape([-1,dims])) / ds))
        self._dt = dt
        next_states = np.zeros([num_actions, num_states, dims])
        print('Time step, dt = {}'.format(dt))

        for act_idx, action in enumerate(all_actions):
            next_states = dyn.integrate(all_states, action, dt)
            temp = (next_states - s_min) / ds
            # Lower grid idx of interpolating hypercube.
            grid_idx_min = np.floor(temp).astype(int)
            # Interp weight for the lower idx of each dimension 
            alpha = 1 - (temp - grid_idx_min)
            for shift in interp_region:
                interp_grid_idx = np.minimum(np.maximum(grid_idx_min + shift,
                                             0), np.array(num_nodes) - 1)
                interp_weight = np.prod(alpha * (1 - shift) +
                                        (1 - alpha) * shift, axis=1)
                temp = list(state_to_idx(interp_grid_idx, np.array(num_nodes)))

                p_trans[act_idx, range(num_states), temp] +=\
                    interp_weight
        super().__init__(num_nodes, p_trans, all_states,
                 all_actions, gamma)

    def gradient(self):
        v_grid = self.v_opt.reshape(self._num_nodes)
        grad = np.zeros([self._dims] + list(self._num_nodes))
        ds = self._ds
        dim_ord = list(range(0,self._dims))
        grad_mag = 0
        for k in range(0,self._dims):
            dim_ord[0] = k
            dim_ord[k] = 0
            v_grid_tran = np.transpose(v_grid, tuple(dim_ord))
            temp = np.zeros(v_grid_tran.shape)
            temp[0,:] = (v_grid_tran[1,:] - v_grid_tran[0,:]) / ds[k]
            temp[-1,:] = (v_grid_tran[-1,:] - v_grid_tran[-2,:]) / ds[k]
            temp[1:-1,:] = (v_grid_tran[2:,:] - v_grid_tran[:-2,:]) / 2 / ds[k]
            grad[k] = np.transpose(temp, tuple(dim_ord));
            dim_ord[0] = 0
            dim_ord[k] = k
            grad_mag += grad[k]**2
        grad_mag = grad_mag ** 0.5
        return grad, grad_mag
    
    def _state_to_idx(self, states):
        """Takes states and returns indices."""
        states_disc= (states - self._s_lims[0, :]) / self._ds
        return state_to_idx(states_disc, self._num_nodes)
    
    @property 
    def ds(self):
        """Return dimension of the state space."""
        return self._ds

    @property
    def dynamics(self):
        return self._dynamics

    @dynamics.setter
    def dynamics(self, dynamics):
        s_min = self._s_lims[0, :]
        s_max = self._s_lims[1, :]
        a_min = self._a_lims[0, :]
        a_max = self._a_lims[1, :]
        dims = len(self._num_nodes)
        deriv = np.zeros([self._num_actions, self._num_states, dims])
        p_trans = np.zeros([self._num_actions, self._num_states,
                            self._num_states])

        # All states (actions) as grid indices
        state_axes = [np.arange(N_d) for N_d in self._num_nodes]
        all_states = cartesian(state_axes) * self._ds + s_min

        action_axes = [np.arange(N_d) for N_d in self._num_nodes_a]
        all_actions = cartesian(action_axes) * self._da + a_min

        # Construct interpolation weights (transition probabilities)
        dyn = Dynamics(dynamics, self._num_nodes.size)  # dynamics model

        # Hypercube defining interpolation region
        interp_axes = [np.array([0, 1]) for d in range(dims)]
        interp_region = cartesian(interp_axes).astype(int)

        for act_idx, action in enumerate(all_actions):
            deriv[act_idx, :, :] = dyn.deriv(all_states, action)

        # State moves at most one grid cell in any dimension over one time
        # step.

        dt = (1.0 / np.amax(np.abs(deriv.reshape([-1, dims])) / self._ds))
        self._dt = dt
        next_states = np.zeros([self._num_actions, self._num_states, dims])
        print('Time step, dt = {}'.format(dt))

        for act_idx, action in enumerate(all_actions):
            next_states = dyn.integrate(all_states, action, dt)
            temp = (next_states - s_min) / self._ds
            # Lower grid idx of interpolating hypercube.
            grid_idx_min = np.floor(temp).astype(int)
            # Interp weight for the lower idx of each dimension
            alpha = 1 - (temp - grid_idx_min)
            for shift in interp_region:
                interp_grid_idx = np.minimum(np.maximum(grid_idx_min + shift,
                                                        0),
                                             np.array(self._num_nodes) - 1)
                interp_weight = np.prod(alpha * (1 - shift) +
                                        (1 - alpha) * shift, axis=1)
                temp = list(state_to_idx(interp_grid_idx, np.array(
                    self._num_nodes)))

                p_trans[act_idx, range(self._num_states), temp] += \
                    interp_weight

        self._p_trans = p_trans

class Avoid(GridWorld):
    """MDP to compute safe set for a specified avoid set.

    The value function corresponds to the minimum distance to the avoid 
    set. 

    Args:
        num_nodes (1D np array): Number of nodes in each state dimension.
        s_lims(2d np array): Range on the states. 
            First (second) column lower (upper) bound for each state dim.
        num_nodes_a (1D np array): Number of nodes in each action dimension.
        a_lims(2d np array): Range on the actions. 
            First (second) column lower (upper) bound for each action dim.
        dynamics(function): Continous dynamics model.
             Function takes in state (x) and action(a), and returns
             the state derivative (x_dot) 
        avoid_func(function): A signed distance function to the avoid set.
    """
    def __init__ (self, num_nodes, s_lims, num_nodes_a, a_lims=None, dynamics=None, avoid_func=None, lamb=0):
        
        super().__init__(num_nodes, s_lims, num_nodes_a, a_lims, dynamics)
        dt = self._dt
        self._gamma = np.exp(-dt * lamb)
        self._reward = avoid_func(self._all_states)

