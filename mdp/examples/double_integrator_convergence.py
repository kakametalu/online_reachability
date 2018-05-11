import numpy as np
from mdp.dynamics import double_integrator
from mdp.signed_distance import hypercube_int, hypersphere_ext, union, set_minus
from mdp.grid_world_ext import Avoid
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # Grid parameters
    num_nodes = np.array([161, 161])
    s_lims = np.array([[-1, -5], [5, 5]])  # state space limits
    num_nodes_a = np.array([80])
    a_lims = np.array([[0], [1]])  # action/control limits

    # Dynamical system (double integrator model)
    grav = 9.81  # gravity
    sys_params = {}  # parameters of dynamical system
    max_u = 0.2 * grav
    min_u = -0.2 * grav
    sys_params['max_u'] = max_u
    sys_params['min_u'] = min_u
    dynamics = partial(double_integrator, **sys_params)

    # Construct avoid region, system should stay within hypercube
    cube_lims = np.array([[0, -3], [4, 3]])
    nose_lims = np.array([[1.75, -0.25], [2.25, 0.25]])
    lips_lims = np.array([[0, -3], [4, -2.5]])
    left_eye_center = np.array([0.75,2])
    right_eye_center = np.array([3.25,2])
    radius = 0.25
    avoid_func = lambda x: hypercube_int(x, cube_lims=cube_lims)
    face_func = lambda x: hypersphere_ext(x, np.array([2, 0]), 4)
    lip_func = lambda x: -hypercube_int(x, lips_lims)
    nose_func = lambda x: -hypercube_int(x, nose_lims)
    left_eye_func = lambda x: hypersphere_ext(x, left_eye_center, radius)
    right_eye_func = lambda x: hypersphere_ext(x, right_eye_center, radius)

    union_func = union(shapes = [left_eye_func, right_eye_func])

    # With discounting
    lamb = 0.1  # lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, dynamics,
                     avoid_func, lamb=lamb)

    all_states = my_world._all_states_c
    init_vi = set_minus(face_func, union_func)(all_states)
    #init_vi = avoid_func(all_states)
    # Compute value function and policy
    #v_opt, pi_opt = my_world.v_pi_opt(v=init_vi,method='pi')

    '''
    # Without discounting
    lamb = 0.0  # lambda
    my_world_wd = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, dynamics,
                     avoid_func, lamb=lamb)

    # Compute value function and policy
    v_opt_wd, pi_opt_wd = my_world_wd.v_pi_opt(v_opt=init_vi, method='pi')
    '''
    # Computing analytic safe set
    s_min = s_lims[0]
    s_max = s_lims[1]
    x = range(my_world.num_nodes[0]) * my_world.ds[0] + s_min[0]
    y = range(my_world.num_nodes[1]) * my_world.ds[1] + s_min[1]
    u_lims = cube_lims[1]
    l_lims = cube_lims[0]
    '''
    analytic_1 = [min((-2 * min_u * (u_lims[0] - min(x_e, u_lims[0]))) ** 0.5,
                      u_lims[1]) for x_e in x]
    analytic_2 = [max(-(2 * max_u * (max(x_e, 0))) ** 0.5, l_lims[1]) for x_e in
                  x]

    # level sets to be visualized
    L = np.max(my_world.reward)
    tau = 1
    c = L * (1 - np.exp(-lamb * tau))  # under approximation level curve
    v_func_conts = [0]

    # Plot contours of value function
    plt.figure(1)
    CS = plt.contour(x, y, v_opt.reshape(num_nodes).T, levels=v_func_conts)
    CS_wd = plt.contour(x, y, v_opt_wd.reshape(num_nodes).T, levels=v_func_conts)
    


    CS.collections[0].set_label('Discounted min-dist')
    CS_wd.collections[0].set_label('Without discounting')
    plt.plot(x, analytic_1, 'b-.', label='Analytic Safe Set')
    plt.plot(x, analytic_2, 'b-.')
    plt.title('Convergence Contours')
    plt.legend()
    '''
    plt.figure(1)
    CS = plt.contour(x, y, init_vi.reshape(num_nodes).T, levels=[0])
    CS.collections[0].set_label('init v')

    plt.title('Convergence Contours')
    plt.legend()

    plt.pause(100)