# This module provides an example of path planning in a grid world

import numpy as np
from mdp.grid_world import ReachAvoid, rectangles_to_states
from mdp.discrete_transition_models import random_walk_model

if __name__ == "__main__":
    # Number of nodes per dimension
    num_nodes = [50, 50]

    # Creating action probabilities
    # f (intended), l (left of intended), r (right of intended), b(backward)
    # s(stay)
    f ,l, r, b, s = [0.6, 0.1, 0.1, 0.1, 0.1]
    action_probs = np.array([[0.0, 0.25, 0.25, 0.25, 0.25],
                             [s, f, l, b, r],
                             [s, r, f, l, b],
                             [s, b, r, f, l],
                             [s, l, b, r, f]])
    
    p_trans, all_states, all_actions = random_walk_model(num_nodes,
    	                                                 action_probs)

    # Define obstacles and goal
    obs_regions = np.array([[[20, 40],[30, 40]],
                          [[40, 45],[10, 40]]])

    obs_states = rectangles_to_states(obs_regions)

    goal_regions = np.array([[[45, 50],[45, 50]]])
    goal_states = rectangles_to_states(goal_regions)


    # Create MDP
    my_world = ReachAvoid(num_nodes, avoid=obs_states, reach=goal_states,
    	                  gamma=1.0, p_trans=p_trans, all_states=all_states,
    	                  all_actions=all_actions)

    # Simulation parameters
    start_state = np.array([9,0])
    horizon = 50
    policy = np.ones([np.prod(num_nodes)]).astype(int) * 2



    # Compute optimal value function and policy.
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')
    my_world.visualize_policy(pi_opt) # Visualize policy.
    #my_world.visualize_v_func(v_opt) # Visualize value function.

    print(pi_opt[450])
    my_world.simulate(start_state, horizon, visualize_on=True, deterministic=True) # Simulate.