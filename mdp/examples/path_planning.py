# This module provides an example of path planning in a grid world

import numpy as np
from mdp.grid_world import ReachGoal

    
if __name__ == "__main__":
    # Number of nodes per dimension
    num_nodes = [10, 10]

    # Creating action probabilities
    # f (intended), l (left of intended), r (right of intended), s(stay)
    f ,l, r, s = [0.7, 0.1, 0.1, 0.1]
    action_probs = np.array([[0.7, 0.1, 0.1, 0.1, 0.1],
                             [s, f, l, 0, r],
                             [s, r, f, l, 0],
                             [s, 0, r, f, l],
                             [s, l, 0, r, f]])

    # Define obstacles
    obstacles = np.array([[[2,6],[8,9]],
                          [[6,9],[1,5]]])

    start_state = np.array([9,0])
    goal = np.array([9,9])
    horizon = 50
    policy = np.ones([np.prod(num_nodes)]).astype(int) * 2

    my_world = ReachGoal(num_nodes, obstacles=obstacles, goal=goal,
    	                 action_probs=action_probs)

    # Compute optimal value function and policy.
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')
    my_world.visualize_policy(pi_opt) # Visualize policy.
    my_world.simulate(start_state, horizon, visualize_on=True) # Simulate.