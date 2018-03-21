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
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.draw import ellipsoid


if __name__ == "__main__":

    # Grid parameters
    num_nodes = np.array([21, 21, 21])
    s_lims = np.array([[-6, -10, 0],[15, 10, 2*np.pi]]) #state space limits
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
    avoid_func = lambda x: hypersphere_ext(x, center=cen, radius=rad,
                                           dims= dist_dims)
       

    # Make MDP
    lamb = 0 #lambda
    my_world = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, num_nodes_d, 
    	             d_lims, dynamics=dynamics, avoid_func=avoid_func,
    	             lamb=lamb, sparse=True)

    lamb_2 = 0.01 #lambda
    my_world_2 = Avoid(num_nodes, s_lims, num_nodes_a, a_lims, num_nodes_d, 
                     d_lims, dynamics=dynamics, avoid_func=avoid_func,
                     lamb=lamb_2, sparse=True)

    reward = my_world.reward
    value_v, _ =  my_world.v_pi_opt(method='vi')
    value_z, _ =  my_world_2.v_pi_opt(method='vi')
    grid = my_world._all_states_c

    # Generates level curves for target (reward) and value_function
    verts_n_r, faces_r, _, _ = measure.marching_cubes(reward.reshape(num_nodes), 0)
    verts_r = verts_n_r / num_nodes * (s_lims[1,:] - s_lims[0,:]) + s_lims[0,:]

    verts_n_v, faces_v, _, _ = measure.marching_cubes(value_v.reshape(num_nodes), 0)
    verts_v = verts_n_v / num_nodes * (s_lims[1,:] - s_lims[0,:]) + s_lims[0,:]

    verts_n_z, faces_z, _, _ = measure.marching_cubes(value_z.reshape(num_nodes), 0)
    verts_z = verts_n_z / num_nodes * (s_lims[1,:] - s_lims[0,:]) + s_lims[0,:]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(verts_r[:, 0], verts_r[:,1], faces_r, verts_r[:, 2],
                           lw=1, color='b')
    ax.plot_trisurf(verts_v[:, 0], verts_v[:,1], faces_v, verts_v[:, 2],
                          lw=1, color='r', alpha=.2)
    ax.plot_trisurf(verts_z[:, 0], verts_z[:,1], faces_z, verts_z[:, 2],
                           lw=1, color='g', alpha=0.5, 
                           )

    ax.plot([0],[0], 'b', label='Target Set')
    ax.plot([0],[0], 'r', label='V(x) Zero Level Set')
    ax.plot([0],[0], 'g', label='Z(x) Zero Level Set ' + r'$lambda$'+'={}'.format(lamb_2))

    ax.set_xlim(s_lims[0,0], s_lims[1,0])  
    ax.set_ylim(s_lims[0,1], s_lims[1,1])    
    ax.set_zlim(s_lims[0,2], s_lims[1,2]) 

    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("x_3")
    ax.legend()
    plt.tight_layout()
    plt.show()
