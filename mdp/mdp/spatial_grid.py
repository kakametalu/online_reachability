# This module contains classes that extend the GridWorld based classes in
# grid_world.py to approximations of continous spaces. 

import numpy as np
from mdp.grid_world import GridWorld as GridWorldDiscrete, state_to_idx
from mdp.dynamics import Dynamics, double_integrator
from mdp.signed_distance import dist_hypercube_int
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D


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
        gamma 
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
            dim_ord[0] = k;
            dim_ord[k] = 0;
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

class Avoid(GridWorld):
    """MDP to compute safe set for a specified avoid set.

    Args:



    """
    def __init__ (self, num_nodes, s_lims, num_nodes_a, a_lims=None, dynamics=None, avoid_func=None):
        
        super().__init__(num_nodes, s_lims, num_nodes_a, 
                         a_lims, dynamics)        
        impl_avoid = avoid_func(self._all_states)
        avoid = (impl_avoid <= 0).nonzero()[0]
        self._avoid_set = set(avoid)
        super().add_abs(avoid)
        dt = self._dt
        self._gamma = np.exp(-dt)
        self._reward[:,:] = 1.0 - np.exp(-dt)
        self._reward[avoid, :] = 0.0

# def visualize_v_func(self, v_func = None, contours=None):
#     """ Visualize contour plot of value function.

#         Args:
#             v_func(1D np array): Value function to be visualized.
#                 Size is number of states.
#         """

#         assert(self._dims==2 or self._dims==1),\
#             "Can only visualize value functions for 1D and 2D grids."

#         plt.figure(figsize=(8, 8))
#         if self._dims ==1:
#             plt.plot(v_func)
#         else:
#             s_min = self._s_lims[:,0]
#             s_max = self._s_lims[:,1]

#             x = range(self._num_nodes[0]) * self._ds[0] + s_min[0] 
#             y = range(self._num_nodes[1]) * self._ds[1] + s_min[1]
#             # z = [min((-2*self.a_lims[0]*(self.u_lims[0]-min(x_e,self.u_lims[0])))**0.5,self.u_lims[1]) for x_e in x]

#             # z2 = [max(-(2*self.a_lims[1]*(max(x_e,0)))**0.5, self.l_lims[1]) for x_e in x]
#             #y=-sqrt(max(vs{1},0)*(uMax1-g)*2);

#             if contours is None:
#                 plt.contour(x, y, v_func.reshape(self._num_nodes).T)
#             else:
#                 plt.contour(x, y, v_func.reshape(self._num_nodes).T, contours)
#                 # fig = plt.figure()
#                 # ax = fig.gca(projection='3d')
#                 # X, Y = np.meshgrid(x, y)

#                 # ax.plot_surface(X, Y, v_func.reshape(self._num_nodes).T)
#                 # plt.plot(x,z,'b-.')
#                 # plt.plot(x,z2,'r-.')
    
#             # plt.savefig('value_function.png')
#             plt.pause(100) 


if __name__ == "__main__":
    
    num_nodes = np.array([41, 41])
    s_lims = np.array([[-1,-5],[5,5]])
    num_nodes_a = np.array([2])
    a_lims = np.array([[-0.2],[0.2]]) * 9.81
    dynamics = double_integrator 
    cube_lims = np.array([[0, -3], [4, 3]])

    k_func = lambda x: dist_hypercube_int(x, cube_lims=cube_lims)
 
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, dynamics, k_func)
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')
    exit_time = np.linspace(1.0, 1.5, 10)
    dt = my_world._dt
    gamma = my_world._gamma
    Ns = [time/dt for time in exit_time]
    contours=[(1-gamma**(N+1)) * dt / (1 - gamma) for N in Ns]

    ttr = (np.log(1 - v_opt * (1 - gamma)/dt)/np.log(gamma) - 1) * dt
    ttr_2 = -np.log(1 -  v_opt)
    contours_2 = 1 - np.exp(-exit_time)
    grad, grad_mag = my_world.gradient()
    contours_3=np.linspace(0, 3.0, 20) 
    print(np.min(grad_mag[:]))
    print(np.max(grad_mag[:]))

    # Plot contours
    s_min = s_lims[0]
    s_max = s_lims[1]
    x = range(my_world.num_nodes[0]) * my_world.ds[0] + s_min[0] 
    y = range(my_world.num_nodes[1]) * my_world.ds[1] + s_min[1]
    print(s_min)
    plt.contour(x, y, v_opt.reshape(num_nodes).T)

    u_lims = cube_lims[1]
    l_lims = cube_lims[0]

    z = [min((-2*a_lims[0]*(u_lims[0]-min(x_e, u_lims[0])))**0.5,u_lims[1]) for x_e in x]

    z2 = [max(-(2*a_lims[1]*(max(x_e,0)))**0.5, l_lims[1]) for x_e in x]
    plt.plot(x,z,'b-.')
    plt.plot(x,z2,'r-.')
    plt.pause(100) 

