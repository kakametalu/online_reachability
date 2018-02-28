# This script explores updating the value function online

import numpy as np
from mdp.dynamics import double_integrator
from mdp.signed_distance import hypercube_int
from mdp.grid_world_ext import Avoid
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    
    # Grid parameters
    num_nodes = np.array([41, 41])
    s_lims = np.array([[-1,-5],[5,5]]) #state space limits
    num_nodes_a = np.array([2])
    a_lims = np.array([[0],[1]]) #action/control limits

    #Dynamical system (double integrator model)
    gain_i = 1.0 # gain of initial model
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    max_u = gain_i * grav
    min_u = -gain_i * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)
    
    #Final model gain
    gain_f = 0.2

    # Construct avoid region, system should stay within hypercube 
    cube_lims = np.array([[0, -3], [4, 3]])
    avoid_func = lambda x: hypercube_int(x, cube_lims=cube_lims)
 
    # Make MDP
    lamb = 0.1 #lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a,
                     a_lims, dynamics, avoid_func, lamb=lamb)

    # Compute value function and policy
    v_opt_i, _ = my_world.v_pi_opt()


    # Model estimation and value function update
    
    horizon=101
    for k in range(horizon):
        print("Online Iteration {}".format(k))
        alpha = k/horizon
        gain = gain_f * alpha + gain_i * (1 - alpha)
        sys_params = {}
        sys_params['max_u'] = grav * gain
        sys_params['min_u'] = -grav * gain
        dynamics = partial(double_integrator, **sys_params)
        my_world.dynamics = dynamics
        my_world.update()
    
    # max_u = gain_f * grav
    # min_u = -gain_f * grav
    # sys_params['max_u'] = max_u
    # sys_params['min_u'] = min_u
    # dynamics = partial(double_integrator, **sys_params)
    # my_world.dynamics = dynamics

    v_opt_ws, _ = my_world.v_pi_opt() # current estimate of val function

    # Compute final val function from warm start estimate above
    v_opt_1, _ = my_world.v_pi_opt(V=v_opt_ws, force_run=True)

    # Compute final val function from signed distance func
    v_opt_2, _ = my_world.v_pi_opt(force_run=True)

    # Look to see which of the above two function calls finished faster



    # Computing anaylytic safe set (Model 2)
    s_min = s_lims[0]
    s_max = s_lims[1]
    x = range(my_world.num_nodes[0]) * my_world.ds[0] + s_min[0] 
    y = range(my_world.num_nodes[1]) * my_world.ds[1] + s_min[1]
    u_lims = cube_lims[1]
    l_lims = cube_lims[0]
    
    analytic_1 = [min((-2*min_u*(u_lims[0]-min(x_e, u_lims[0])))**0.5,
                  u_lims[1]) for x_e in x]
    analytic_2 = [max(-(2*max_u*(max(x_e,0)))**0.5, l_lims[1]) for x_e in x]

    # Plot contours of value functions
    plt.figure(1)
    CS = plt.contour(x, y, v_opt_1.reshape(num_nodes).T, levels=[0], colors='g')
    CS.collections[0].set_label('V_$\\lambda$ Final zero level')

    CS_i = plt.contour(x, y, v_opt_i.reshape(num_nodes).T, levels=[0], colors='r')
    CS_i.collections[0].set_label('V_$\\lambda$ Initial zero level')

    plt.plot(x, analytic_1,'b-.', label='Analytic Safe Set')
    plt.plot(x, analytic_2,'b-.')
    plt.title('Value Function Contours')
    plt.legend()
 
    plt.pause(100) 