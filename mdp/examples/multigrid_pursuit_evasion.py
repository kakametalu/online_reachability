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
    num_nodes_c = np.array([node/2, node/2, node/2]).astype(int) + 1 
    num_nodes_f = np.array([node, node, node]).astype(int) + 1 
    
    ang_u_f = 2*np.pi * (1 - 1/(num_nodes_f[2]))
    s_lims_f = np.array([[-6, -10, 0],[20, 10, ang_u_f]]) #state space limits    
    eps = 10**-6
    s_lims_c = s_lims_f + np.array([[-eps], [eps]]) #state space limits

    num_nodes_a = np.array([2])
    a_lims = np.array([[-1],[1]]) #action/control limits
    num_nodes_d = np.array([2])
    d_lims = np.array([[-1],[1]]) #action/control limits

    #Dynamical system (pursuit evasion)
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
           
    # Make MDP
    lamb = 0.01 #lambda
    my_world_c = Avoid(num_nodes_c, s_lims_c, num_nodes_a, a_lims, 
                       num_nodes_d, d_lims, dynamics=dynamics, 
                       avoid_func=avoid_func, lamb=lamb, sparse=True)


    my_world_f = Avoid(num_nodes_f, s_lims_f, num_nodes_a, a_lims, 
                       num_nodes_d, d_lims, dynamics=dynamics, 
                       avoid_func=avoid_func, lamb=lamb, sparse=True)
    
    grid_f = my_world_f._all_states_c

    # Compute value function on coarse
    t_start = time.time()
    value_c, _ =  my_world_c.v_pi_opt(method='vi')
    t_coarse = time.time() - t_start

    # Compute value function on fine with warm start value_c 
    t_start = time.time()
    warm_start = my_world_c.interp_grid(value_c, grid_f)
    value_f_warm, _ =  my_world_f.v_pi_opt(method='vi', V=warm_start)
    t_fine_warm = time.time() - t_start

    # Warm start from value_c 
    t_start = time.time()
    value_f, _ =  my_world_f.v_pi_opt(method='vi',force_run=True)
    t_fine = time.time() - t_start


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

    pickle.dump(results, open('multigrid_{}.pkl'.format(node), 'wb'))
