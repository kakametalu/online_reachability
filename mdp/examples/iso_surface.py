import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mdp.dynamics import double_integrator, dubins_car, pursuit_evasion
from mdp.signed_distance import hypercube_int, hypersphere_ext, union,\
    intersect
from mdp.grid_world_ext import Avoid
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# def fun(x, y, z):
#     return cos(x) + cos(y) + cos(z)

# x, y, z = pi*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
# vol = fun(x, y, z)
# verts, faces, _, _ = measure.marching_cubes(vol, 0, spacing=(0.1, 0.1, 0.1))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
#                 cmap='Spectral', lw=1)
# plt.show()



# Three dimensional dubins car model. Avoid set is a circle in 2D euclidean 
# space. This script shows how to take grid slices.




if __name__ == "__main__":

    # Grid parameters
    num_nodes = np.array([121, 121, 121])
    s_lims = np.array([[-10,-10,0],[10,10,2*np.pi]]) #state space limits
    num_nodes_a = np.array([2])
    a_lims = np.array([[-1],[1]]) #action/control limits
    num_nodes_d = np.array([2])
    d_lims = np.array([[-1],[1]]) #action/control limits

    #Dynamical system (dubins_car)
    sys_params = {} # parameters of dynamical system
    sys_params['v_u'] = 5
    sys_params['v_d'] = 5
    dynamics = partial(pursuit_evasion, **sys_params)
    



    # Construct avoid region (circle in 2D euclidean space)
    cen =  np.array([0,0]) #center
    rad = 5.0 # radius
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
    lamb = 0.1 #lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, num_nodes_d, 
    	             d_lims, dynamics=dynamics, avoid_func=avoid_func,
    	             lamb=lamb, sparse=True)
    grid = my_world._all_states_c
    grid_axes = my_world.axes
    value, _ =  my_world.v_pi_opt(method='vi')
    reward =  my_world.reward


    # verts, faces, _, _ = measure.marching_cubes(value.reshape(num_nodes), 0,
    #                                             spacing=(0.1, 0.1, 0.1))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
    #                 lw=1)
    # plt.show()

    # x =  new_axes[0]
    # y =  new_axes[1]
    

    # Take slice of value and reward grids
    dim_fix = [2] # dims to be held fixed
    val_fix = [np.pi/2] # value along fixed dimensions

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