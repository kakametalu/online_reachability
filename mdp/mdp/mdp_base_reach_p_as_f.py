import numpy as np
from scipy.sparse import csr_matrix, linalg
from copy import deepcopy
from itertools import product
import time
from cvxopt import matrix, solvers

class TransitionModel_f(object):
    """Dictionary containing the state transition probabilities for an MDP.

    Attributes:
        _num_states (uint): Number of states.
        _num_actions (uint): Number of actions.
        _f_trans(function): By default it is a self transition function

    Args:
        num_states (uint): Number of states.
        num_actions (uint): Number of actions.
        f_trans (function): By default it is a self transition function
    """

    def __init__(self,num_states, num_actions, f_trans=None):
        """Initialize TransitionModel object."""

        if f_trans is None:
            f_trans = lambda state, action: [state], np.array([1.0])

        self._f_trans = f_trans
        self._num_states = num_states
        self._num_actions = num_actions

    def __getitem__(self, state_action):
        """Takes state and action and returns (support, probs, exp rewards).

        Args:
            state_action (tuple of uints): State index and action index.
        Returns:
            support (1D np array): Possible next transition states.
            probs (1D np array): Transition probabilities.
        """
        
        state, action = state_action
        assert (state < self._num_states), "State index is out of range."
        assert (action < self._num_actions), "Action index is out of range."

        support, probs = self._f_trans(state, action)
        return support, probs
    
    @property
    def f_trans(self):
        return self._f_trans
    
    @f_trans.setter
    def f_trans(self, f_trans):
        self._f_trans = f_trans

    @property
    def num_states(self):
        return self._num_states
    @property
    def num_actions(self):
        return self._num_actions


class MDP_f(TransitionModel_f):
    """MDP tuple with states, actions, rewards, and transition function.

    The trajectory reward is a sum of discounted rewards. The rewards may
    be undiscounted if there are absorbing states with the follwing properties:
        1. Absorbing states can only transition to other absorbing states.
        2. The MDP must eventually transition into an absorbing state.
        3. No reward can be accumulated after an absorbing state is reached.
    

    Attributes:
        _reward (2D np array): Reward function R(s)
            Size is number of states by number of actions.
        _gamma (float): Discount rate in (0 1].
            If gamma is 1, then the reward is undiscounted, and absorbing
            states must be provided.
        _abs_set (list of uints): Set of absorbing states.
        _pi_opt(np array): Optimal action to reach goal from each state.
            Size is total number of states by 1.
        _v_opt(np array): Optimal value (cost to go) from each state.
            Size is total number of states by 1.

    Args:
        num_states (uint): Number of states.
        num_actions(uint): Number of actions.
        _reward (1D np array): Reward function R(s)
            Size is number of states.
        gamma (float): Discount rate between 0 and 1.
            If gamma is 1, then the reward is undiscounted, and absorbing
            states must be provided.
        abs_set (list or set of uints): Set of absorbing states.
        f_trans(function): State transition probabilities is an input
            function
    """

    def __init__(self, num_states, num_actions,
                 reward = None, gamma=None, abs_set=set([]),
                 f_trans=None):
        """Initialize MDP Object."""
        
        gamma_one_abs_none = False # Gamma is 1 and no absorbing state
        if (gamma == 1 and abs_set is None):
            gamma_one_abs_none = True
        assert(gamma_one_abs_none is False),\
            "Absorbing states needed for gamma == 1."

        super().__init__(num_states, num_actions, f_trans)

        # Handling rewards
        if reward is None:
            reward = np.zeros([num_states])
        self._reward = reward

        if gamma is None:
            gamma = 0.95
        self._gamma = gamma

        self._abs_set = set(abs_set)
        self._pi_opt = None
        self._v_opt = None

    def __getitem__(self, state_action):
        """Takes state and action and returns (support, probs, exp rewards).

        Args:
            state_action (tuple of uints): State index and action index.
        Returns:
            support (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
            rewards (float): Reward for state/action pair.
        """
        state, action = state_action
        if state in self._abs_set: 
            support, probs = [[state], [1.0]]
        else:
            support, probs = super().__getitem__(state_action)
        
        reward = self._reward[state]
        return support, probs, reward

    def _policy_backup(self, V, pi):
        """Does one policy back up on the value function."""

        V_out = np.zeros([self.num_states])
        for state in range(self.num_states):
            support, probs, _ = self[state, pi[state]]
            for next_state, p in zip(support, probs):
                V_out[state] += (V[next_state] * p)
        return V_out

    def _bellman_backup(self, V=None):
        """Performs one bellman backup on the value function V."""

        max_reward = np.max(self._reward)
        if V is None:
            V = self._reward 

        ones_vec = np.ones(self.num_states).astype(int)
        pi = ones_vec * 0

        V_out = self._policy_backup(V, pi)
        pi_greedy = pi
        
        if self.num_actions == 1:
            return V_out, pi_greedy
   
        for act in range(1, self.num_actions):
            pi = ones_vec * act
            V_act = self._policy_backup(V, pi)
            pi_greedy = pi_greedy * (V_out >= V_act) +\
                        pi * (V_out < V_act)
            V_out = np.maximum(V_out, V_act)
        V_out = np.minimum((V_out - max_reward) * self._gamma,
         self._reward - max_reward) + max_reward 

        return V_out, pi_greedy

    def _value_iteration(self, V=None, pi=None):
        """Value iteration initialized with initial value function V.
        """        
        if V is None:
            V = self._reward + (np.random.rand(self._num_states)-0.5) * 0
        V_opt = deepcopy(V)
        tol = 10 ** (-3)/2
        err = tol * 2

        count=0
        print("    err (inf norm)")
        while err > tol:
            V_old = deepcopy(V_opt)
            V_opt, pi_opt = self._bellman_backup(V_old)
            err = np.linalg.norm(V_old - V_opt, ord= float('inf'))
            print("%i:  %.6e" %(count,err))
            count += 1
        return V_opt, pi_opt

    def v_pi_opt(self, V=None, pi=None, method='vi',force_run=False):
        """Rerurn optimal value function and policy.
            
            Args:
                V (1D np array): Initial value function.
                    Size is number of states. Only used by value iteration
                    and policy iteration.
                pi (1D np array): Initial policy.
                    Size is number of states. Only used by policy iteration.
                method(string): Method for computing optimal value function.
                    'vi': value iteration, 'pi': policy iteration
            Returns:
                v_opt (1D np array): Optimal value function.
                    Size is number of states.
                pi_opt (1D np array): Optimal policy
                    Size is number of states.
        """

        if self._pi_opt is None or self._v_opt is None or force_run:
            pass
        else:
            return self._v_opt, self._pi_opt

        name = {'vi': 'value iteration',
                'pi': 'policy_iteration'}[method]
        print("Computing optimal value function " + 
              "and policy using {}... ".format(name))
        
        t_start = time.time()
        self._v_opt, self._pi_opt = {'vi': self._value_iteration}[method](V,pi)
        
        print("Done. Elapsed time {}.\n".format(time.time()-t_start))
        return self._v_opt, self._pi_opt

    @property
    def reward(self):
        return self._reward

    @property
    def v_opt(self):
        if self._v_opt is None:
            self.v_pi_opt() 
        return self._v_opt

    @property
    def pi_opt(self):
        if self._pi_opt is None:
            self.v_pi_opt()      
        return self._pi_opt

    @property
    def abs_set(self):
        return self._abs_set

    @abs_set.setter
    def abs_set(self, abs_set, ):
        self._abs_set = abs_set

    def add_abs(self, new_abs):
        """Add new absorbing states.

        Args:
            new_abs (set/list of ints): Additional absorbing states.
        """
        if isinstance(new_abs, set):
            self._abs_set.update(new_abs)
        else:
            self._abs_set.update(set(new_abs))

    
