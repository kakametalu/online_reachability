import numpy as np
from mdp.dynamics import double_integrator
from mdp.signed_distance import dist_hypercube_int
from mdp.grid_world_ext import Avoid
from mdp.grid_world_ext_p_as_f import Avoid_f
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
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    max_u = 0.2 * grav
    min_u = -0.2 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)
    
    # Construct avoid region, system should stay within hypercube 
    cube_lims = np.array([[0, -3], [4, 3]])
    avoid_func = lambda x: dist_hypercube_int(x, cube_lims=cube_lims)
 
    # Make MDP
    lamb = 0.1 #lambda
    my_world = Avoid_f(num_nodes, s_lims, num_nodes_a, a_lims, dynamics,
                     avoid_func, lamb=lamb)

    # Compute value function and policy
    v_opt, pi_opt = my_world.v_pi_opt()

    # Computing anaylytic safe set
    s_min = s_lims[0]
    s_max = s_lims[1]
    x = range(my_world.num_nodes[0]) * my_world.ds[0] + s_min[0] 
    y = range(my_world.num_nodes[1]) * my_world.ds[1] + s_min[1]
    u_lims = cube_lims[1]
    l_lims = cube_lims[0]
    
    analytic_1 = [min((-2*min_u*(u_lims[0]-min(x_e, u_lims[0])))**0.5,
                  u_lims[1]) for x_e in x]
    analytic_2 = [max(-(2*max_u*(max(x_e,0)))**0.5, l_lims[1]) for x_e in x]

    # level sets to be visualized
    L = np.max(my_world.reward)
    tau = 1 
    c = L * (1 - np.exp(-lamb * tau)) #under approximation level curve
    v_func_conts = [0, c]

    # Plot contours of value function
    plt.figure(1)
    CS = plt.contour(x, y, v_opt.reshape(num_nodes).T, levels=v_func_conts)
    
    labels = ['V_$\\lambda$ zero level', 'Safe Set Under-approx']

    for i in range(len(labels)):
        CS.collections[i].set_label(labels[i])
    plt.plot(x, analytic_1,'b-.', label='Analytic Safe Set')
    plt.plot(x, analytic_2,'b-.')
    plt.title('Value Function Contours')
    plt.legend()
 
    plt.pause(100) 