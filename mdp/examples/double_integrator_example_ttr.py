import numpy as np
from mdp.dynamics import double_integrator
from mdp.signed_distance import dist_hypercube_int
from mdp.grid_world_ext import Avoid
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    
    # Grid parameters
    num_nodes = np.array([41, 41])
    s_lims = np.array([[-1,-5],[5,5]])
    num_nodes_a = np.array([2])
    a_lims = np.array([[0],[1]])

    #Dynamical system (double integrator model)
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    max_u = 0.2 * grav
    min_u = -0.2 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)
    
    #Dynamical system_2
    grav = 9.81 # gravity
    sys_params = {} # parameters of dynamical system
    max_u = 0.8 * grav
    min_u = -0.8 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics_2 = partial(double_integrator, **sys_params)

    # Construct avoid region, system should stay within hypercube 
    cube_lims = np.array([[0, -3], [4, 3]])
    avoid_func = lambda x: dist_hypercube_int(x, cube_lims=cube_lims)
 
    # Make MDP
    my_world = Avoid(num_nodes, s_lims, num_nodes_a,
                     a_lims, dynamics, avoid_func)

    # Make MDP
    #my_world_2 = Avoid(num_nodes, s_lims, num_nodes_a,
    #                 a_lims, dynamics_2, avoid_func)
    
    # Compute value function and policy
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')
    #v_opt, pi_opt = my_world_2.v_pi_opt(method='pi',pi=pi_opt)

    # Gradient of value function
    grad, grad_mag = my_world.gradient()

    # Computing anaylytic safe set
    s_min = s_lims[0]
    s_max = s_lims[1]
    x = range(my_world.num_nodes[0]) * my_world.ds[0] + s_min[0] 
    y = range(my_world.num_nodes[1]) * my_world.ds[1] + s_min[1]
    u_lims = cube_lims[1]
    l_lims = cube_lims[0]
    z = [min((-2*min_u*(u_lims[0]-min(x_e, u_lims[0])))**0.5,u_lims[1]) for x_e in x]
    z2 = [max(-(2*max_u*(max(x_e,0)))**0.5, l_lims[1]) for x_e in x]

    exit_time = np.linspace(2.0, 4.0, 10)
    dt = my_world._dt
    gamma = my_world._gamma

    ttr = - np.log(1 -  v_opt)
    v_func_conts = 1 - np.exp(-exit_time)

    # Plot contours of value function
    plt.figure(1)
    CS = plt.contour(x, y, v_opt.reshape(num_nodes).T, levels=v_func_conts)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.plot(x,z,'b-.')
    plt.plot(x,z2,'r-.')
    plt.title('Value Function Contours')
 
    # Plot contours of gradient magnitude
    grad_contours = np.linspace(.7,4.0,10)
    plt.figure(2)
    print(np.max(grad_mag))
    plt.contour(x, y, grad_mag.reshape(num_nodes).T, levels=grad_contours)
    plt.plot(x,z,'b-.')
    plt.plot(x,z2,'r-.')
    plt.title('Gradient (magnitude) Contours')

    plt.pause(100) 