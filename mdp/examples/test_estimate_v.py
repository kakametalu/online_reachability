'''
This module provides an example using
1.nearest_neighbor as a function approximator
2.averaging among neighbors as fucntion approximator
'''

# This module provides an example of path planning in a grid world

import numpy as np
from mdp.grid_world import ReachAvoid, rectangles_to_states
from mdp.discrete_transition_models import random_walk_model
from mdp.function_approximation import nearest_neighbor, averaged_V

def define_action_probs(f, l, r, b, s):
    action_probs = np.array([[0.0, 0.25, 0.25, 0.25, 0.25],
                             [s, f, l, b, r],
                             [s, r, f, l, b],
                             [s, b, r, f, l],
                             [s, l, b, r, f]])
    return action_probs


# Number of nodes per dimension
num_nodes = [50, 50]

# Creating action probabilities
# f (intended), l (left of intended), r (right of intended), b(backward)
# s(stay)

# Define obstacles and goal
obs_regions = np.array([[[20, 40], [30, 40]],
                        [[40, 45], [10, 40]]])

obs_states = rectangles_to_states(obs_regions)

goal_regions = np.array([[[45, 50], [45, 50]]])
goal_states = rectangles_to_states(goal_regions)

lib_nn_V = nearest_neighbor()
lib_avg_V = averaged_V()

for i in range(10):
    x = np.random.rand(5)
    x = x/sum(x)
    f, l, r, b, s = x
    action_probs = define_action_probs(f,l, r, b, s)

    p_trans, all_states, all_actions = random_walk_model(num_nodes,
                                                     action_probs)

    # Create MDP
    my_world = ReachAvoid(num_nodes, avoid=obs_states, reach=goal_states,
                      gamma=1.0, p_trans=p_trans, all_states=all_states,
                      all_actions=all_actions)

    # Compute optimal value function and policy.
    v_opt, pi_opt = my_world.v_pi_opt(method='pi')

    lib_nn_V.insert_new_element(theta=(f,l,r,b,s), V=v_opt)
    lib_avg_V.insert_new_element(theta=(f,l,r,b,s), V=v_opt)




