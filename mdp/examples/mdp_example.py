# This module provides a simple example of how to use the MDP class

import numpy as np
from mdp.mdp_environments import RandomWalk1D
import time	

def end_indicator(s, a, s_next): 
    """Indicator for transitioning to last state.""" 
    if s_next == nS-1 and s_next != s:
        return 1.0
    return 0.0

if __name__ == "__main__":
    
    # Initialize a RandomWalk1D Model
    nS = 10

    absorbing = [0, nS - 1]
    # Direction probabilities
    dir_probs = np.array([[0.2, 0.3, 0.5],
                      [0.1, 0.3, 0.6],
                      [0.4, 0.5, 0.1]])

    # Reward of 1 is given for transitioning into last state.
    my_mdp = RandomWalk1D(nS, dir_probs, end_indicator, gamma=1.0,
                          absorbing=absorbing)

    
    # Run (and time) value iteration.
    t_start = time.time() 
    V_opt, pi_opt = my_mdp.v_opt()
    print("\nValue iteration finished in {} seconds".
    	  format(time.time()- t_start))
    print("Optimal value function:\n {}".format(V_opt))
    print("Optimal policy:\n {}".format(pi_opt))

    
    # Run (and time) policy iteration.
    t_start = time.time() 
    V_opt, pi_opt = my_mdp.v_opt(method='pi')
    print("\nPolicy iteration finished in {} seconds".
    	  format(time.time()- t_start))
    print("Optimal value function:\n {}".format(V_opt))
    print("Optimal policy:\n {}".format(pi_opt))
    
    # Run (and time) linear program.
    # TODO: LP solver has problems for many states. Needs to be fixed. (Kene)
    t_start = time.time() 
    V_opt, pi_opt = my_mdp.v_opt(method='lp')
    print("\nLinear program finished in {} seconds".
    	  format(time.time()- t_start))
    print("Optimal value function:\n {}".format(V_opt))
    print("Optimal policy:\n {}".format(pi_opt))
    