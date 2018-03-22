# This module contains classes that extend the GridWorld based classes in
# grid_world.py to approximations of continous spaces. 

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from mdp.grid_world import GridWorld as GridWorldDiscrete, state_to_idx
from mdp.dynamics import Dynamics
from sklearn.utils.extmath import cartesian
from scipy.sparse import csr_matrix
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
        da2(1d np array): Grid spacing for each action_two dim.

        dt(float): Time step.
    
    Args:
        num_nodes (1D np array): Number of nodes in each state dimension.
        s_lims(2d np array): Range on the states. 
            First (second) column lower (upper) bound for each state dim.
        num_nodes_a (1D np array): Number of nodes in each action dimension.
        a_lims(2d np array): Range on the actions. 
            First (second) column lower (upper) bound for each action dim.
        num_nodes_a2 (1D np array): Number of nodes in each action_two dim.
        a2_lims(2d np array): Range on the disturbances. 
            First (second) column lower (upper) bound for each action dim.
        dynamics(function): Continous dynamics model.
             Function takes in state (x) and action(a), and returns
             the state derivative (x_dot) 
        gamma(float): Discount factor.
    """

    def __init__(self, num_nodes, s_lims, num_nodes_a=None, a_lims=None, 
                 num_nodes_a2=None, a2_lims=None, dynamics=None, gamma=None,
                 sparse=False, angular=None):

        # Preparing grid
        if num_nodes_a is None or a_lims is None:
            num_nodes_a = np.array([1])
            a_lims=np.array([[0],[1]])
        
        if num_nodes_a2 is None or a2_lims is None:
            num_nodes_a2 = np.array([1])
            a2_lims=np.array([[0],[1]])

        self._a_lims = a_lims
        self._s_lims = s_lims
        s_min = s_lims[0, :]
        s_max = s_lims[1, :]
        a_min = a_lims[0, :]
        a_max = a_lims[1, :]
        a2_min = a2_lims[0, :]
        a2_max = a2_lims[1, :]
        eps = 10**(-10) # for numerical stability
        ds = (s_max - s_min)/(num_nodes - 1 + eps)
        da = (a_max - a_min)/(num_nodes_a - 1 + eps) 
        da2 = (a_max - a2_min)/(num_nodes_a2 - 1 + eps) 

        self._ds = ds
        self._da = da
        self._da2 = da2

        self._num_nodes_a = num_nodes_a
        self._num_nodes_a2 = num_nodes_a2

        num_states = np.prod(num_nodes)
        num_actions = np.prod(num_nodes_a)
        num_actions2 = np.prod(num_nodes_a2)

        dims = len(num_nodes)
        deriv = np.zeros([num_actions2, num_actions, num_states, dims])
        p_trans = np.zeros([num_actions2, num_actions, num_states, 
                            num_states])
        # All states (actions) as grid indices
        self._axes = [np.arange(num_nodes[i]) * ds[i] + s_min[i]
                    for i in range(dims)]
        all_states = cartesian(self._axes)
        self._all_states_c = all_states
        
        action_axes = [np.arange(N_d) for N_d in num_nodes_a]
        all_actions = cartesian(action_axes) * da + a_min
        self._all_actions_c = all_actions

        action2_axes = [np.arange(N_d) for N_d in num_nodes_a2]
        all_actions2 = cartesian(action2_axes) * da2 + a2_min
        self._all_actions2_c = all_actions2

        if dynamics is None: # Just exit after grid has been built
            self._dt = 1
            super().__init__(num_nodes, p_trans, all_states, 
                             all_actions, all_actions2, gamma)
            return

        # Construct interpolation weights (transition probabilities)
        dyn = Dynamics(dynamics, num_nodes.size, angular=angular) # dynamics model


        # Hypercube defining interpolation region
        interp_axes = [np.array([0,1]) for d in range(dims)]
        interp_region = cartesian(interp_axes).astype(int)
        

        for act2_idx, action2 in enumerate(all_actions2):
            for act_idx, action in enumerate(all_actions):
                deriv[act2_idx, act_idx,:,:] = dyn.deriv(all_states, action,
                                                         action2)

        # State moves at most one grid cell in any dimension over one time
        # step.

        dt = (1.0 / np.amax(np.abs(deriv.reshape([-1,dims])) / ds))
        self._dt = dt 
        next_states = np.zeros([num_actions, num_states, dims])
        if sparse:
            p_trans = {}
        t_start = time.time()
        for act2_idx, action2 in enumerate(all_actions2):
            for act_idx, action in enumerate(all_actions):
                next_states = dyn.integrate(dt, all_states, action, action2)
                temp = (next_states - s_min) / ds
                # Lower grid idx of interpolating hypercube.
                grid_idx_min = np.floor(temp).astype(int)
                # Interp weight for the lower idx of each dimension 
                alpha = 1 - (temp - grid_idx_min)
                if sparse:
                    p_trans_indiv = csr_matrix((num_states, num_states))
                for shift in interp_region:
                    interp_grid_idx = np.minimum(np.maximum(grid_idx_min +
                                                            shift, 0),
                                                 np.array(num_nodes) - 1)

                    interp_weight = np.prod(alpha * (1 - shift) +
                                            (1 - alpha) * shift, axis=1)
                    temp = list(state_to_idx(interp_grid_idx, 
                                             np.array(num_nodes)))
                    if sparse:
                        shift_vals = csr_matrix((interp_weight, (range(
                            num_states), temp)), shape=(num_states, num_states))
                        p_trans_indiv = p_trans_indiv + shift_vals
                    else:
                        p_trans[act2_idx, act_idx, range(num_states), temp] += \
                        interp_weight
                if sparse:
                    p_trans[(act2_idx, act_idx)] = p_trans_indiv
        print('Time to convert to csr', time.time() - t_start)
        super().__init__(num_nodes, p_trans, all_states,
                 all_actions, all_actions2, gamma)

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

    def interp_grid(self, values, points):
        """Returns the interpolation of values at points.

            Args:
                values(1D np array): The values of the function on the grid.
                   Size is equal to the number of states in the grid. Values
                   should correspond to the rows of self._all_states_c.
                points (2D np array): Points to be interpolated.
                    Shape is number of points by self._dims.
            
            Returns:
                new_values(1D np array): Values at points.
                   Size is equal to the number of points. 
        """ 
        gi =RegularGridInterpolator(self._axes,
                                    values.reshape(self._num_nodes),
                                    bounds_error=False, fill_value=1000)
        new_values = gi(points)
        return new_values

    def slice_grid(self, values, dim_fix, val_fix):
        """Returns slice of the grid.

            Args: 
                values(1D np array): The values of the function on the grid.
                    Size is equal to the number of states in the grid. Values
                    should correspond to the rows of self._all_states_c.
                dim_fix (1D array of ints): The dims that are being held fixed.
                val_fix (1D array of floats): Value for fixed dims.
                    The order of val_fix should coincide with dim_fix.
            Return:
                new_values(1D np array): A grid slice of values.
                new_grid_shape (1D np array ints): Shape of new grid.
                new_axes_flat(list of 1D np arrays): Axes of remaining dims.
        """
        
        new_axes = []
        new_axes_flat = []
        new_grid_shape = np.zeros([self.dims - len(dim_fix)])
        dim_fix_count = 0
        grid_shape_count = 0

        for k in range(self.dims):
            if k not in dim_fix:
                new_axes.append(self.axes[k])
                new_axes_flat.append(self.axes[k])
                new_grid_shape[grid_shape_count] = self._num_nodes[k]
                grid_shape_count += 1

            else:
                new_axes.append(np.array([val_fix[dim_fix_count]]).astype(float))
                dim_fix_count += 1
        
        points = cartesian(new_axes)
        new_values = self.interp_grid(values, points)
        return new_values, new_grid_shape.astype(int), new_axes_flat

    @property 
    def ds(self):
        """Return dimension of the state space."""
        return self._ds

    @property
    def axes(self):
        """Return list of axes values for each dimension."""
        return self._axes
    
    @property
    def dynamics(self):
        return self._dynamics

    @dynamics.setter
    def dynamics(self, dynamics):
        s_min = self._s_lims[0, :]
        ds = self._ds
        num_actions = self._num_nodes_a
        num_states = self.num_states
        dims = len(self._num_nodes)
        deriv = np.zeros([self._num_actions, self._num_states, dims])
        p_trans = np.zeros([self._num_actions2, self._num_actions, 
                            self._num_states, self._num_states])

        all_states = self._all_states_c
        all_actions = self._all_actions_c
        all_actions2 = self._all_actions2_c

        # Construct interpolation weights (transition probabilities)
        dyn = Dynamics(dynamics, self._num_nodes.size)  # dynamics model

        # Hypercube defining interpolation region
        interp_axes = [np.array([0, 1]) for d in range(dims)]
        interp_region = cartesian(interp_axes).astype(int)

        for act2_idx, action2 in enumerate(all_actions2):
            for act_idx, action in enumerate(all_actions):
                deriv[act2_idx, act_idx,:,:] = dyn.deriv(all_states, action,
                                                         action2)

        # State moves at most one grid cell in any dimension over one time
        # step.

        dt = (1.0 / np.amax(np.abs(deriv.reshape([-1,dims])) / ds))
        self._dt = dt 
        next_states = np.zeros([num_actions, num_states, dims])
        
        for act2_idx, action in enumerate(all_actions2):
            for act_idx, action in enumerate(all_actions):
                next_states = dyn.integrate(dt, all_states, action, action2)
                temp = (next_states - s_min) / ds
                # Lower grid idx of interpolating hypercube.
                grid_idx_min = np.floor(temp).astype(int)
                # Interp weight for the lower idx of each dimension 
                alpha = 1 - (temp - grid_idx_min)
                for shift in interp_region:
                    interp_grid_idx = np.minimum(np.maximum(grid_idx_min +
                                                            shift, 0),
                                                 np.array(num_nodes) - 1)

                    interp_weight = np.prod(alpha * (1 - shift) +
                                            (1 - alpha) * shift, axis=1)
                    temp = list(state_to_idx(interp_grid_idx, 
                                             np.array(num_nodes)))

                    p_trans[act2_idx, act_idx, range(num_states), temp] += \
                        interp_weight

        if self.sparse:
            p_trans_new = np.zeros((self._num_nodes_a2,self._num_nodes_a))
            for a2, a1 in zip(self._all_actions2_c, self._all_actions_c):
                p_trans_new[a2][a1] = csr_matrix(p_trans[a2, a1])
            p_trans = p_trans_new

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
    def __init__ (self, num_nodes, s_lims, num_nodes_a=None, a_lims=None, 
                  num_nodes_a2=None, a2_lims=None, dynamics=None, 
                  avoid_func=None, lamb=0, sparse=False,angular=None):
        
        super().__init__(num_nodes, s_lims, num_nodes_a, a_lims,
                         num_nodes_a2, a2_lims, dynamics, sparse=sparse,
                         angular=angular)
        dt = self._dt
        self._gamma = np.exp(-dt * lamb)

        if lamb == 0:
            self.tol = (1 - np.exp(-dt * .00001)) * 10**(-1)
        else:
            self.tol = (1-self._gamma) * 10**(-1)
        
        self.tol = 10**-3
        self._reward = avoid_func(self._all_states)

