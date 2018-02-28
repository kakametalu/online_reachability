# Three dimensional dubins car model. Avoid set is a circle in 2D euclidean 
# space. This script shows how to take grid slices.

import numpy as np
from mdp.dynamics import double_integrator, dubins_car
from mdp.signed_distance import hypercube_int, hypersphere_ext, union,\
    intersect
from mdp.grid_world_ext import Avoid
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

if __name__ == "__main__":

    # Grid parameters
    num_nodes = np.array([21, 21, 21]) 
    s_lims = np.array([[-2,-2,-np.pi/2],[2,2,np.pi/2]]) #state space limits
    num_nodes_a = np.array([1, 3])
    a_lims = np.array([[1, -np.pi/10],[1, np.pi/10]]) #action/control limits

    #Dynamical system (dubins_car)
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    sys_params['L'] = 1
    sys_params['max_speed'] = 1
    dynamics = partial(dubins_car, **sys_params)
    



    # Construct avoid region (circle in 2D euclidean space)
    cen =  np.array([0,0]) #center
    rad = 0.5 # radius
    dist_dims =  [0, 1] # dimensions contributing to distance computation
    avoid_func = lambda x: hypersphere_ext(x, center=cen,radius=rad,
                                           dims= dist_dims)
       
    # Example for taking unions of sets  
    # cen_1 =  np.array([0,0])
    # rad_1 = 0.5
    # avoid_func_1 = lambda x: hypersphere_ext(x, center=cen_1,radius=rad_1,
    #                                        dims=[0, 1])
    # cen_2 =  np.array([1,1])
    # rad_2 = 1
    # avoid_func_2 = lambda x: hypersphere_ext(x, center=cen_2,radius=rad_2,
    #                                        dims=[0, 1])
    # avoid_func = intersect([avoid_func_1, avoid_func_2])
    
    # Make MDP
    lamb = 0.0001 #lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a,
                     a_lims, dynamics, avoid_func, lamb=lamb)
    grid = my_world._all_states_c
    grid_axes = my_world.axes
    value, _ =  my_world.v_pi_opt(method='pi')
    reward =  my_world.reward

    # Take slice of value and reward grids
    dim_fix = [2] # dims to be held fixed
    val_fix = [0] # value along fixed dimensions

    val_slice, new_shape, new_axes = my_world.slice_grid(value, dim_fix,
                                                         val_fix)
    
    reward_slice, _, _ = my_world.slice_grid(reward, dim_fix, val_fix)    

    x =  new_axes[0]
    y =  new_axes[1]
    
    # Plot contours of value and reward
    plt.figure(1)
    CS_1 = plt.contour(x, y, val_slice.reshape(new_shape).T,
                       levels=[0.00001], colors='b')
    CS_1.collections[0].set_label('Reachable Set')

    CS_2 = plt.contour(x, y, reward_slice.reshape(new_shape).T,
                     levels=[0.00001], colors='r')
    CS_2.collections[0].set_label('Target Set')

    plt.title('Vehicle Angle State = {}'.format(val_fix[0]))
    plt.legend()
 
    plt.pause(100) 