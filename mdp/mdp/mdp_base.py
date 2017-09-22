import numpy as np
from scipy.optimize import linprog
from copy import deepcopy

class TransitionProbabilities(object):
    """Dictionary containing the state transition probabilities for an MDP.

    Attributes:
        _trans_dict(dict): Dictionary containing transition probabilities.
            Takes state and action index, and returns tuple of np arrays, where
            each list has a state index, and transition probability, e.g.
            ([state_0, state_1], [0.3, 0.7]) = _trans_dict[state_0, action_0].
        _num_states (uint): Number of states.
        _num_actions (uint): Number of actions.
        _trans_tensor(3d np array): State transition probabilities in a tensor.
            Tensor dimension is num_actions by num_states by num_states.
            Usage: _trans_tensor[action, state, next_state]

    Args:
        num_states (uint): Number of states.
        num_actions (uint): Number of actions.
    """

    def __init__(self,num_states, num_actions):
        """Initialize TransitionProbabilities object."""

        self._trans_dict = {}
        self._trans_tensor = np.zeros([num_actions, num_states, num_states])
        self._num_states = num_states
        self._num_actions = num_actions

    def add_transition(self, state, action, next_states, probs,
                       skip_prob_assert=False):
        """Add new transition.

        Args:
            state (uint): State index.
            action (uint): Action index.
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
            skip_prob_assert (bool): Skip probability assertion.
               User intentionally has specified probs to not sum to one.
        """
        assert (state < self._num_states),\
            "State index {} out of range.".format(state)
        assert (action < self._num_actions), "Action index out of range."
        if skip_prob_assert is False:
            assert (np.sum(probs) == 1.0),\
                "Total prob = {}".format(np.sum(probs))
        for next_state in next_states:
            assert (next_state < self._num_states),\
                "Next state {} index out of range.".format(next_state)

        self._trans_dict.update({(state, action): (next_states, probs)})
        self._trans_tensor[action, state, next_states] = probs

    def __getitem__(self, state_action):
        """Takes state and action and returns (next_states, probs, exp rewards).

        Args:
            state_action (tuple of uints): State index and action index.
        Returns:
            next_states (1D np array): Possible next transition states.
            probs (1D np array): Transition probabilities.
        """
        
        state, action = state_action
        assert (state < self._num_states), "State index is out of range."
        assert (action < self._num_actions), "Action index is out of range."

        # If no transition for state/action then use default.
        default = ([state], [0.0])
        (next_states, probs) = self._trans_dict.get((state, action), default)

        return next_states, probs
    
    @property
    def trans_tensor(self):
        return self._trans_tensor
    @property
    def num_states(self):
        return self._num_states
    @property
    def num_actions(self):
        return self._num_actions

def zero_reward(s, a, s_next):
    """Reward function that returns zero."""
    return 0.0

class MDP(TransitionProbabilities):
    """MDP tuple with states, actions, rewards, and transition function.

    The trajectory reward is a sum of discounted rewards. The rewards may
    be undiscounted if there are absorbing states with the follwing properties:
        1. Absorbing states can only transition to other absorbing states.
        2. The MDP must eventually transition into an absorbing state.
        3. No reward can be accumulated after an absorbing state is reached.
    

    Attributes:
        _reward_func (func): Scalar reward function.
            Usage reward = _reward_func(state, action, next_state).
        _gamma (float): Discount rate in (0 1].
            If gamma is 1, then the reward is undiscounted, and absorbing
            states must be provided.
        _expR (np array): Expected immediate reward for state action pair.
            Shape is num_states by num_actions.
        absorbing (list of uints): Absorbing state indices.

    Args:
        num_states (uint): Number of states.
        num_actions(uint): Number of actions.
        _reward_func (func): Scalar reward function.
            Usage reward = _reward_func(state, action, next_state).
        gamma (float): Discount rate between 0 and 1.
            If gamma is 1, then the reward is undiscounted, and absorbing
            states must be provided.
        _absorbing (list of uints): Absorbing state indices.
    """

    def __init__(self, num_states, num_actions, reward_func = None,
                 gamma=None, absorbing=None):
        """Initialize MDP Object."""
        
        if reward_func is None:
            reward_func = zero_reward

        if gamma is None:
            gamma = 0.95
        
        gamma_one_abs_none = False # Gamma is 1 and no absorbing state
        if (gamma == 1 and absorbing is None):
            gamma_one_abs_none = True
        assert(gamma_one_abs_none is False),\
            "Absorbing states needed for gamma = 1."

        super().__init__(num_states, num_actions)
        self._reward_func = reward_func
        self._gamma = gamma
        self._expR = np.zeros([num_states, num_actions])
        self._absorbing = absorbing


    def add_transition(self, state, action, next_states, probs):
        """Add new transition and expected reward.

        Args:
            state (uint): State index.
            action (uint): Action index.
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
        """

        # Absorbing states will be treated as transitioning out of the
        # state space in the undiscounted case. This is useful for computing
        # the value function via a linear program.
        
        spa = False # Skip probability assertion
        probs_copy = deepcopy(probs) # Pass by copy since it may be modified.
        if state in self._absorbing and self._gamma ==1.0:
            probs_copy *= 0
            spa = True

        super().add_transition(state, action, next_states, probs_copy,spa)
        self._expR[state, action] = np.sum([self.reward(state, action, x[0])
                                            * x[1]for x in zip(next_states,
                                                               probs_copy)])
    def __getitem__(self, state_action):
        """Takes state and action and returns (next_states, probs, exp rewards).

        Args:
            state_action (tuple of uints): State index and action index.
        Returns:
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
            rewards (float): Reward for state/action pair.
        """
        state, action = state_action
        next_states, probs = super().__getitem__(state_action)
        exp_reward = self._expR[state, action]
        return next_states, probs, exp_reward

    def reward(self, state, action, next_state):
        """Return reward."""

        return self._reward_func(state, action, next_state)

    def _policy_backup(self, V, pi):
        """Does one policy back up on the value function."""
        V_out = np.zeros(self.num_states)
        for state in range(self.num_states):
            next_support, ps, exp_r= self[state, pi[state]]
            V_out[state] = exp_r
            for next_state, p in zip(next_support, ps):
                try:
                    V_out[state] += (V[next_state] * p) * self._gamma
                except TypeError:
                    print(V)
        return V_out

    def _bellman_backup(self, V=None):
        """Performs one bellman backup on the value function V."""
        if V is None:
            V = np.zeros(self.num_states)

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

    def v_opt(self, V=None, method='vi'):
        """Rerurn optimal value function and policy.
            
            Args:
                V (1D np array): Initial value function.
                    Size is number of states.
                method(string): Method for computing optimal value function.
                    'vi': value iteration, 'pi': policy iteration, 'lp':
                    linear_programming
            Returns:
                V_opt (1D np array): Optimal value function.
                    Size is number of states.
                pi_opt (1D np array): Optimal policy
                    Size is number of states.
        """
        V_opt, pi_opt = {'vi': self._value_iteration,
                         'pi': self._policy_iteration,
                         'lp': self._linear_program}[method](V)
        
        return V_opt, pi_opt

    def _value_iteration(self, V=None):
        if V is None:
            V = np.zeros(self.num_states)
        V_opt = deepcopy(V)
        tol = 10 ** (-20)
        err = tol * 2

        while err > tol:
            V_old = deepcopy(V_opt)
            V_opt, pi_opt = self._bellman_backup(V_old)
            err = np.linalg.norm(V_old - V_opt, ord= float('inf'))

        return V_opt, pi_opt

    def _policy_iteration(self, V=None):
        """Policy iteration initialized with value function V.

            Args:
                V (1D np array): Initial value function.
                    Size is number of states.
            Returns:
                V_opt (1D np array): Optimal value function.
                    Size is number of states.
                pi_opt (1D np array): Optimal policy
                    Size is number of states.
        """
        if V is None:
            V = np.zeros(self.num_states)
        
        V_pi, pi = self._bellman_backup(V)
        tol = 10 ** (-16)

        while True:
            err = tol * 2
            while err > tol:
                V_old = V_pi
                V_pi = self._policy_backup(V_old, pi)
                err = np.linalg.norm(V_old - V_pi, ord= float('inf'))
            V_opt, pi_opt = self._bellman_backup(V_pi)
            
            # Check for policy convergence.
            if (pi_opt == pi).all(): 
                break
            pi = pi_opt
            V_pi = V_opt
        return V_opt, pi_opt

    def _linear_program(self, V=None):
        c = np.ones(self.num_states)
        eye_tensor = np.zeros([self.num_actions, self.num_states,
                               self.num_states])
        eye_tensor[:, range(self.num_states), range(self.num_states)] = 1.0
        A_ineq = np.concatenate(list(self._gamma * self._trans_tensor -
                                     eye_tensor))
        b_ineq = - self._expR.flatten('F')
        A_eq = np.zeros([2, self.num_states])
        A_eq[0,0] = 1
        A_eq[1,self.num_states-1] = 1
        b_eq = np.zeros([2,1])
        
        def my_callback(xk,**kwargs):
            #print("current solution:\n {}".format(xk))
            return
            
        output = linprog(c, A_ineq, b_ineq, callback=my_callback)
        V_opt = output['x']
        print(output['status'])
        print(output['success'])
        print(output['nit'])
        print(output['message'])
        if np.isnan(V_opt).any():
            print("found a nan here too")
            return
        V_opt, pi_opt = self._value_iteration(V_opt)

        return V_opt, pi_opt
