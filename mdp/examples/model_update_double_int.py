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
    node = 320
    num_nodes = np.array([node, node]).astype(int) + 1 
        
    s_lims = np.array([[-1,-5],[5,5]]) #state space limits

    num_nodes_a = np.array([2])
    a_lims = np.array([[0],[1]]) #action/control limits   


    #Dynamical systems (double integrator model: nominal, light, heavy)
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system (nominal)
    max_u = 0.2 * grav
    min_u = -0.2 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)

    sys_params_l = {} # parameters of dynamical system (light)
    max_u_l = 0.1 * grav
    min_u_l = -0.1 * grav
    sys_params_l['max_u'] = max_u_l
    sys_params_l['min_u'] = min_u_l
    dynamics_l = partial(double_integrator, **sys_params_l)

    sys_params_h = {} # parameters of dynamical system (light)
    max_u_h = 0.4 * grav
    min_u_h = -0.4 * grav
    sys_params_h['max_u'] = max_u_h
    sys_params_h['min_u'] = min_u_h
    dynamics_h = partial(double_integrator, **sys_params_h)

    # Construct avoid region, system should stay within hypercube 
    cube_lims = np.array([[0, -3], [4, 3]])
    avoid_func = lambda x: hypercube_int(x, cube_lims=cube_lims)
 
     # Make MDP
    lamb = 0.1 #lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, dynamics=dynamics,
                     avoid_func=avoid_func, lamb=lamb, sparse=True)

    # light model
    my_world_l = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, 
                       dynamics=dynamics_l, avoid_func=avoid_func, lamb=lamb,
                       sparse=True)
   
    # heavy model
    my_world_h = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, 
                       dynamics=dynamics_h, avoid_func=avoid_func, lamb=lamb,
                       sparse=True)

    # Compute nominal value func
    t_start = time.time()
    value_n,_ = my_world.v_pi_opt(method='vi')
    t_n = time.time() - t_start

    #Compute value func for light model
    t_start = time.time()
    value_l_warm, _ = my_world_l.v_pi_opt(method='vi', V=value_n)
    t_l_warm = time.time() - t_start
    
    t_start = time.time()
    value_l, _ = my_world_l.v_pi_opt(method='vi',force_run=True)
    t_l = time.time() - t_start

    #Compute value func for heavy model
    t_start = time.time()
    value_h_warm, _ = my_world_h.v_pi_opt(method='vi', V=value_n)
    t_h_warm = time.time() - t_start
    

    t_start = time.time()
    value_h, _ = my_world_h.v_pi_opt(method='vi',force_run=True)
    t_h = time.time() - t_start

    print("Time for nominal: {}".format(t_n))
    print("Time for light model with warm start: {}".format(t_l_warm))
    print("Time for light model: {}".format(t_l))
    print("Time for heavy model with warm start: {}".format(t_h_warm))
    print("Time for heavy model: {}".format(t_h))

    results = {'value_n' : value_n,
               'value_a_warm' : value_l_warm,
               'value_a' : value_l,
               'value_d_warm' : value_h_warm,
               'value_d' : value_h,
               't_a' : t_l,
               't_d' : t_h,
               't_a_warm' : t_l_warm,
               't_d_warm' : t_h_warm,
               }

    pickle.dump(results, open('model_update_double_int_{}.pkl'.format(node),
                              'wb'))
