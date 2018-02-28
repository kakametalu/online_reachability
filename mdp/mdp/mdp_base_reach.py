import numpy as np
from scipy.sparse import csr_matrix, linalg
from copy import deepcopy
from itertools import product
import time
from cvxopt import matrix, solvers

class TransitionModel(object):
    """Dictionary containing the state transition probabilities for an MDP.

    Attributes:
        _num_states (uint): Number of states.
        _num_actions (uint): Number of actions.
        _p_trans(3d np array): State transition probabilities in a tensor.
            Tensor dimension is num_actions by num_states by num_states.
            Usage: _p_trans[action, state, next_state]

    Args:
        num_states (uint): Number of states.
        num_actions (uint): Number of actions.
        p_trans(3d np array): State transition probabilities in a tensor.
            Tensor dimension is num_actions by num_states by num_states.
            Usage: _p_trans[action, state, next_state]
    """

    def __init__(self,num_states, num_actions, p_trans=None):
        """Initialize TransitionModel object."""

        if p_trans is None:
            p_trans = np.zeros([num_actions, num_states, num_states])

        p_trans_shape = (num_actions, num_states, num_states)
        assert (p_trans.shape == p_trans_shape),\
            "Transition tensor should have shape {}.".format(p_trans_shape)
        self._p_trans = p_trans
        self._num_states = num_states
        self._num_actions = num_actions

    def add_transition(self, state, action, support, probs,
                       skip_prob_assert=False):
        """Add new transition.

        Args:
            state (uint): State index.
            action (uint): Action index.
            support (1D np array): Possible next transition states.
            probs (1D np array): Transition probabilities.
        """
        assert (state < self._num_states),\
            "State index {} out of range.".format(state)
        assert (action < self._num_actions), "Action index out of range."
        assert (np.sum(probs) == 1.0),\
            "Total prob = {}".format(np.sum(probs))
        for next_state in support:
            assert (next_state < self._num_states),\
                "Next state {} index out of range.".format(next_state)
        self._p_trans[action, state, support] = probs

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

        support = self._p_trans[int(action), int(state)].nonzero()[0]
        probs = self._p_trans[action, state, support]

        if len(support) == 0:
            support =[state]
            probs = np.array([1.0])
        return support, probs
    
    @property
    def p_trans(self):
        return self._p_trans
    
    @p_trans.setter
    def p_trans(self, p_trans):
        assert (p_trans.shape == self._p_trans.shape),\
            "Transition tensor should have shape {}.".format(self._ptrans.shape)
        # TODO(AKA): check that tensor consist of stochastic matrices.
        self._p_trans = p_trans

    @property
    def num_states(self):
        return self._num_states
    @property
    def num_actions(self):
        return self._num_actions


class MDP(TransitionModel):
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
        p_trans(3d np array): State transition probabilities in a tensor.
            Tensor dimension is num_actions by num_states by num_states.
            Usage: _p_trans[action, state, next_state]
    """

    def __init__(self, num_states, num_actions,
                 reward = None, gamma=None, abs_set=set([]),
                 p_trans=None):
        """Initialize MDP Object."""
        
        gamma_one_abs_none = False # Gamma is 1 and no absorbing state
        if (gamma == 1 and abs_set is None):
            gamma_one_abs_none = True
        assert(gamma_one_abs_none is False),\
            "Absorbing states needed for gamma == 1."

        super().__init__(num_states, num_actions, p_trans)

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
        max_reward = np.max(self._reward)
        nS = self.num_states
        p_pi = self._p_trans[pi, range(nS), :]
        V_out = p_pi.dot(V)
        V_out = np.minimum((V_out - max_reward) * self._gamma,
         self._reward - max_reward) + max_reward 
        return V_out

    def _bellman_backup(self, V=None):
        """Performs one bellman backup on the value function V."""

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


        return V_out, pi_greedy

    def _value_iteration(self, V=None, pi=None, one_step=False):
        """Value iteration initialized with initial value function V.
        """        
        if V is None:
            V = self._reward
        V_opt = deepcopy(V)
        tol = 10 ** (-3)/2
        err = tol * 2

        count=0
        print("    err (inf norm)")
        while err > tol:
            V_old = deepcopy(V_opt)
            V_opt, pi_opt = self._bellman_backup(V_old)
            err = np.linalg.norm(V_old - V_opt, ord= float('inf'))
            if one_step:
                break
            print("%i:  %.6e" %(count,err))
            count += 1
        return V_opt, pi_opt
    def _policy_iteration(self, V=None, pi=None):
        """Policy iteration initialized with initial value function V."""
        
        nS = self.num_states
        if V is None:
            V = self._reward
        
        if pi is None:
            V_pi, pi = self._bellman_backup(V)
        else:
            V_pi = V

        tol = 10 ** (-3)/2
        
        print("    err (inf norm)")
        while True:
            err_in = tol * 2
            V_old_out = deepcopy(V_pi)
            count=0
            while err_in > tol:
                V_old_in = deepcopy(V_pi)
                V_pi = self._policy_backup(V_old_in, pi)
                err_in = np.linalg.norm(V_old_in - V_pi, ord= float('inf'))
                #print(err_in)

            V_opt, pi_opt = self._bellman_backup(V_pi)
            err_out = np.linalg.norm(V_opt - V_old_out, ord= float('inf'))
            print("%i:  %.6e" %(count, err_out))
            count += 1
            # Check for policy convergence.
            if (pi_opt == pi).all() or err_out<tol: 
                break
            pi = pi_opt
            V_pi = V_opt
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
        self._v_opt, self._pi_opt = {'vi': self._value_iteration,
                                     'pi': self._policy_iteration}\
                                     [method](V,pi)
        
        print("Done. Elapsed time {}.\n".format(time.time()-t_start))
        return self._v_opt, self._pi_opt
    
    def update(self):
        """Update the value function by applying one bellman update."""

        self._v_opt, self._pi_opt = self._bellman_backup(self._v_opt)




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

    
