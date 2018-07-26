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
import time 
import pickle

if __name__ == "__main__":

    # Grid parameters
    node = 40
    num_nodes_c = np.array([node/4, node/4]).astype(int) + 1 
    num_nodes_f = np.array([node, node]).astype(int) + 1 
    
    s_lims_f = np.array([[-1,-4],[5,4]]) #state space limit
    eps = 10**-6
    s_lims_c = s_lims_f + np.array([[-eps], [eps]]) #state space limits

    num_nodes_a = np.array([2])
    a_lims = np.array([[0],[1]]) #action/control limits

    #Dynamical system (double integrator model)
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    max_u = 0.2 * grav
    min_u = -0.2 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)
    

   
    # Construct avoid region, system should stay within hypercube 
    cube_lims = np.array([[0, -3], [4, 3]])
    avoid_func = lambda x: hypercube_int(x, cube_lims=cube_lims)
 
    # Make MDP
    lamb = 0.1 #lambda
    my_world_c = Avoid(num_nodes_c, s_lims_c, num_nodes_a, a_lims, 
                       dynamics=dynamics, avoid_func=avoid_func, lamb=lamb,
                       sparse=True)


    my_world_f = Avoid(num_nodes_f, s_lims_f, num_nodes_a, a_lims, 
                       dynamics=dynamics, avoid_func=avoid_func, lamb=lamb,
                       sparse=True)
    
    grid_f = my_world_f._all_states_c

    # Compute value function on coarse
    t_start = time.time()
    value_c, _ =  my_world_c.v_pi_opt(method='vi')
    t_coarse = time.time() - t_start

    # Compute value function on fine
    t_start = time.time()
    value_f, _ =  my_world_f.v_pi_opt(method='vi',force_run=True)
    t_fine = time.time() - t_start

    # Compute value function on fine with warm start value_c 
    warm_start = my_world_c.interp_grid(value_c, grid_f)
    t_start = time.time()
    value_f_warm, _ =  my_world_f.v_pi_opt(method='vi', V=warm_start,
                                           force_run=True)
    t_fine_warm = time.time() - t_start



    # Compute value function on fine with warm start value_c 
    warm_start = my_world_c.interp_grid(value_c, grid_f)
    t_start = time.time()
    value_f_warm, _ =  my_world_f.v_pi_opt(method='vi', V=warm_start,
                                           force_run=True)
    t_fine_warm = time.time() - t_start




    print("Time for coarse: {}".format(t_coarse))
    print("Time for fine w/ warm start: {}".format(t_fine_warm))
    print("Time for multigrid: {}".format(t_fine_warm + t_coarse))
    print("Time for fine: {}".format(t_fine))

    results = {'value_c' : value_c,
               't_coarse' : t_coarse,
               'warm_start' : warm_start,
               't_fine_warm' : t_fine_warm,
               't_fine' : t_fine,
               'value_f_warm' : value_f_warm,
               'value_f' : value_f
               }

    pickle.dump(results, open('multigrid_double_int_{}.pkl'.format(node), 'wb'))
