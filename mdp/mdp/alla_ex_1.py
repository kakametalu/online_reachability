# This module contains classes that extend the GridWorld based classes in
# grid_world.py to approximations of continous spaces. 

import numpy as np
from mdp.grid_world import ReachAvoid, state_to_idx
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import time
from copy import deepcopy

class Dynamics(object):
    def __init__(self, x_dot, dims):
        self._x_dot = x_dot
        self._dims = dims
    
    def deriv(self, states, control):
        assert states.shape[1] == self._dims,"State dimension is incompatible."
        
        return np.array([self._x_dot(state, control) for state in states])
    
    def integrate(self, states, control, t, steps=20):
        """Intergrate ODE using Runge-Kutta 4 scheme."""
        dt = t/steps
        run_t = t/steps
        n_states = deepcopy(states) # next states
    	
        #return n_states + t * self.deriv(n_states, control)
        while  run_t <= t:
            k_1 = self.deriv(n_states, control)
            k_2 = self.deriv(n_states + dt / 2 * k_1, control)
            k_3 = self.deriv(n_states + dt / 2 * k_2, control)
            k_4 = self.deriv(n_states + dt * k_3, control)
            n_states = n_states + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            run_t += dt
        return n_states


def simple_dyn(x, u):
    return np.array(u * (1 - np.abs(x)))

def simple_dyn_2(x, u):
    return np.array([x[1], (1-x[0]**2) * x[1] - x[0] + u[0]])

global dt
global gamma


class ReachAvoidC(ReachAvoid):
    """This class extends the ReachAvoid class to continuous state spaces."""

    def __init__ (self, num_nodes, s_lims, num_nodes_a, a_lims=None, dynamics=None, reward=None):
        global dt
        global gamma 
        lamb = 1.0       
        self.a_lims=a_lims
        my_dyn = Dynamics(dynamics, num_nodes.size)
        
        dims = len(num_nodes)
        s_min = s_lims[0, :]
        s_max = s_lims[1, :]
        a_min = a_lims[0, :]
        a_max = a_lims[1, :]
        ds = (s_max - s_min)/(num_nodes - 1)
        da = (a_max - a_min)/(num_nodes_a - 1)
        
        self._ds = ds
        self._s_min = s_min
        num_states = np.prod(num_nodes)
        num_actions = np.prod(num_nodes_a)

        deriv = np.zeros([num_actions, num_states, dims])
        p_trans = np.zeros([num_actions, num_states, num_states])


        # All states as grid indices
        state_axes = [np.arange(N_d) for N_d in num_nodes]
        all_states = cartesian(state_axes)
        
        action_axes = [np.arange(N_d) for N_d in num_nodes_a]
        all_actions = cartesian(action_axes)

        gamma = 0.99
        super().__init__(num_nodes, p_trans, reach=np.array([]),
                         avoid=np.array([]), gamma=gamma, 
                         all_states=all_states, all_actions=all_actions)
         
        all_states_c = all_states * ds + s_min
        all_actions_c = all_actions * da + a_min

        # Hypercube defining interpolation region
        interp_axes = [np.array([0,1]) for d in range(dims)]
        interp_region = cartesian(interp_axes).astype(int)

        for act_idx, action_c in enumerate(all_actions_c):
            deriv[act_idx,:,:] = my_dyn.deriv(all_states_c, action_c)

        # State moves at most one grid cell in any dimension over one time
        # step.
        dt = (1.0 / np.amax(np.abs(deriv.reshape([-1,dims])) / ds)) * 0.001
        #dt = ds/2
        #dt = (1.0 / np.amax(np.sum(np.abs(deriv.reshape([-1,dims])) / ds, axis=1)))
        gamma =np.exp(-lamb * dt)
        self._gamma= gamma
        self.next_c = np.zeros([num_actions, num_states, dims])
        self.all_states_c=all_states_c
        print('dt = {}'.format(dt))
        for act_idx, action_c in enumerate(all_actions_c):
            next_c = my_dyn.integrate(all_states_c, action_c, dt)
            #next_c = np.minimum( np.maximum(next_c, s_min) , s_max)
            self.next_c[act_idx] = next_c
            temp = (next_c - s_min) / ds
            temp2 = (all_states_c - s_min) / ds

            # Lower grid idx of interpolating hypercube.
            grid_idx_min = np.floor(temp).astype(int)

            # Interp weight for the lower idx of each dimension 
            alpha = 1 - (temp - grid_idx_min)

            sum_weight=0
            for shift in interp_region:
                interp_grid_idx = np.minimum(np.maximum(grid_idx_min + shift,
                                             0), np.array(num_nodes) - 1)
                interp_weight = np.prod(alpha * (1 - shift) +
                                        (1 - alpha) * shift, axis=1)
                temp = list(state_to_idx(interp_grid_idx, np.array(num_nodes)))
                self._p_trans[act_idx, range(self._num_states), temp] +=\
                    interp_weight
                sum_weight += interp_weight
        self.l_lims = np.array([0, -3])
        self.u_lims = np.array([4, 3])
                        
        self._reward[:,:] = -3 * (1 - np.abs(all_states_c)) *dt
        # self._reward[:,:] = (all_states[:,0]**2 + all_states[:,1]**2).reshape([num_states,-1])  * dt

    def visualize_policy(self, policy=None):
        """Visualize the policy.
        
        Args:
            policy(1D np array): Policy to be visualized.
                Size is number of states.
        """
 
        assert(self._dims==2),\
            "Can only visualize policies for 2D grids."

        plt.figure(figsize=(8, 8))
        # plt.axis([-0.5, self._num_nodes[0] - 0.5, -0.5,
        #           self._num_nodes[1] - 0.5])
        plt.ion()
        
        # Symbols for each action
        act_symbol = {0:'o', 3:'$\u2192$', 1:'$\u2191$',
                      4:'$\u2193$', 2:'$\u2193$'}

        state_action = np.concatenate([self.all_states_c,
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
        
        x = range(self._num_nodes[0]) * self._ds[0] +self._s_min[0] 
        plt.title("Optimal Policy", fontsize=20)
        plt.savefig('optimal_policy.png')
        plt.pause(40) # Plot will last three seconds

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
            x = range(self._num_nodes[0]) * self._ds[0] + self._s_min[0] 
            plt.plot(x, v_func)
            z = [-3/2*(x_e+1)*(x_e<0) - 3/2*(1-x_e)*(x_e>=0) for x_e in x]
            plt.plot(x,z,'r-.') 
        else:
            if contours is None:
                x = range(self._num_nodes[0]) * self._ds[0] + self._s_min[0] 
                y = range(self._num_nodes[1]) * self._ds[1] + self._s_min[1] 
                plt.contour(x, y, v_func.reshape(self._num_nodes).T)
            else:
                plt.contour(x, y, v_func.reshape(self._num_nodes).T, contours)

        # plt.savefig('value_function.png')
        plt.pause(100) 



if __name__ == "__main__":

    # num_nodes = np.array([81])
    # s_lims = np.array([[-1],[1]])
    # num_nodes_a = np.array([2])
    # a_lims = np.array([[-1],[1]])
    # dynamics = simple_dyn   
    # my_world = ReachAvoidC(num_nodes, s_lims, num_nodes_a, a_lims, dynamics)
    # v_opt, pi_opt = my_world.v_pi_opt(method='pi')


    num_nodes = np.array([81, 81])
    s_lims = np.array([[-2, -2],[2, 2]])
    num_nodes_a = np.array([2])
    a_lims = np.array([[-1],[1]])
    dynamics = simple_dyn_2  
    my_world = ReachAvoidC(num_nodes, s_lims, num_nodes_a, a_lims, dynamics)
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')


    my_world.visualize_v_func(v_opt) 





