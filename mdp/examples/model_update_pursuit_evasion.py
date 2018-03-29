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
    num_nodes = np.array([node, node, node]).astype(int) + 1 
        
    ang_u = 2*np.pi * (1 - 1/(num_nodes[2]))
    s_lims = np.array([[-6, -10, 0],[20, 10, ang_u]]) #state space limits

    num_nodes_a = np.array([2])
    a_lims = np.array([[-1],[1]]) #action/control limits
    a_lims_s = a_lims * 1.5
    num_nodes_d = np.array([2])
    d_lims = np.array([[-1],[1]]) #action/control limits
    d_lims_s = d_lims * 1.5

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
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, 
                     num_nodes_d, d_lims, dynamics=dynamics, 
                     avoid_func=avoid_func, lamb=lamb, sparse=True,angular=[2])

    # Evader has advantage
    my_world_a = Avoid(num_nodes, s_lims, num_nodes_a, a_lims_s, 
                       num_nodes_d, d_lims, dynamics=dynamics, 
                       avoid_func=avoid_func, lamb=lamb, sparse=True,angular=[2])
   
    # Pursuer has advantage
    my_world_d = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, 
                       num_nodes_d, d_lims_s, dynamics=dynamics, 
                       avoid_func=avoid_func, lamb=lamb, sparse=True, angular=[2]) 

    # Compute nominal value func
    t_start = time.time()
    value_n,_ = my_world.v_pi_opt(method='vi')
    t_n = time.time() - t_start

    #Compute value func for evader advantage
    t_start = time.time()
    value_a_warm, _ = my_world_a.v_pi_opt(method='vi', V=value_n)
    t_a_warm = time.time() - t_start
    
    t_start = time.time()
    value_a, _ = my_world_a.v_pi_opt(method='vi',force_run=True)
    t_a = time.time() - t_start

    #Compute value func for pursuer advantage
    t_start = time.time()
    value_d_warm, _ = my_world_d.v_pi_opt(method='vi', V=value_n)
    t_d_warm = time.time() - t_start
    

    t_start = time.time()
    value_d, _ = my_world_d.v_pi_opt(method='vi',force_run=True)
    t_d = time.time() - t_start

    print("Time for nominal: {}".format(t_n))
    print("Time for evader advantage with warm start: {}".format(t_a_warm))
    print("Time for evader advantage: {}".format(t_a))
    print("Time for pursuer advantage with warm start: {}".format(t_d_warm))
    print("Time for pursuer advantage: {}".format(t_d))

    results = {'value_n' : value_n,
               'value_a_warm' : value_a_warm,
               'value_a' : value_a,
               'value_d_warm' : value_d_warm,
               'value_d' : value_d,
               't_a' : t_a,
               't_d' : t_d,
               't_a_warm' : t_a_warm,
               't_d_warm' : t_d_warm,
               }

    pickle.dump(results, open('model_update_{}.pkl'.format(node), 'wb'))
