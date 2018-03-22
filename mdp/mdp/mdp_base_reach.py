import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, rand
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

    def __init__(self,num_states, num_actions, num_actions2=1, p_trans=None):
        """Initialize TransitionModel object."""

        if p_trans is None:
            p_trans = np.zeros([num_actions2, num_actions, num_states,
                                num_states])
        p_trans_shape = (num_actions2, num_actions, num_states, num_states)
        if isinstance(p_trans[0, 0], csr_matrix):
            self.sparse= True
            #assert (p_trans.shape == p_trans_shape[:2]) and (p_trans[0,
            #    0].shape == p_trans_shape[2:])
        else:
            self.sparse=False
            assert (p_trans.shape == p_trans_shape),\
            "Transition tensor should have shape {}.".format(p_trans_shape)
        self._p_trans = p_trans
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_actions2 = num_actions2

    def add_transition(self, state, action, action2, support, probs,
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
        assert (action2 < self._num_actions2), "Action index out of range."
        assert (np.sum(probs) == 1.0),\
            "Total prob = {}".format(np.sum(probs))
        for next_state in support:
            assert (next_state < self._num_states),\
                "Next state {} index out of range.".format(next_state)
        if self.sparse:
            support_probs = np.zeros(self._num_states)
            support_probs[support] = probs
            m = self._p_trans.tolil()
            m[action2, action][state] = lil_matrix(support_probs)
            self._p_trans = m.tocsr()
        else:
            self._p_trans[action2, action, state, support] = probs

    def __getitem__(self, state_actions):
        """Takes state and action and returns (support, probs, exp rewards).

        Args:
            state_actions (tuple of uints): State, action, action2 indexes.
        Returns:
            support (1D np array): Possible next transition states.
            probs (1D np array): Transition probabilities.
        """
        
        state, action, action2 = state_actions
        assert (state < self._num_states), "State index is out of range."
        assert (action < self._num_actions), "Action index is out of range."
        assert (action2 < self._num_actions2), "Action2 index is out of range."


        if self.sparse:
            probs = self._p_trans[int(action2), int(action)][int(
                state)].tocsr()
        else:
            probs = self._p_trans[int(action2), int(action), int(state)]
        support = probs.nonzero()[0]

        if len(support) == 0:
            support =[state]
            probs = np.array([1.0])
        return support, probs
    
    @property
    def p_trans(self):
        return self._p_trans
    
    @p_trans.setter
    def p_trans(self, p_trans):
        if self.sparse:
            assert (p_trans.shape == self._p_trans.shape) and \
                   (p_trans[0, 0].shape == self._p_trans[0, 0].shape)
        else:
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

    @property
    def num_actions2(self):
        return self._num_actions2

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

    def __init__(self, num_states, num_actions, num_actions2,
                 reward = None, gamma=None, abs_set=set([]),
                 p_trans=None):
        """Initialize MDP Object."""
        
        gamma_one_abs_none = False # Gamma is 1 and no absorbing state
        if (gamma == 1 and abs_set is None):
            gamma_one_abs_none = True
        assert(gamma_one_abs_none is False),\
            "Absorbing states needed for gamma == 1."

        super().__init__(num_states, num_actions, num_actions2, p_trans)

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
        self.tol = 1e-2

    def __getitem__(self, state_actions):
        """Takes state and action and returns (support, probs, exp rewards).

        Args:
            state_action (tuple of uints): State, action, action2, indexes.
        Returns:
            support (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
            rewards (float): Reward for state/action pair.
        """
        state, action, action2 = state_actions
        if state in self._abs_set: 
            support, probs = [[state], [1.0]]
        else:
            support, probs = super().__getitem__(state_actions)
        
        reward = self._reward[state]
        return support, probs, reward

    def _policy_backup(self, V, pi=None, converge=False):
        """Does one policy back up on the value function."""
        max_reward = np.max(self._reward)
        nS = self.num_states
        if pi is not None:
            act_pi = pi[:, 0]
            act2_pi = pi[:, 1]
        else:
            act2_pi = range(self.num_actions2)
            act_pi = range(self.num_actions)
        if self.sparse:
            t_pi_start = time.time()
            if pi is not None:
                p_pi = csr_matrix((nS, nS))
                for p in np.unique(pi, axis=0):
                    n_states = np.where(np.all(pi == p, axis=1))
                    V_act = self._p_trans[p[1], p[0]]
                    p_pi[n_states] = V_act[n_states]
            t_pi = time.time() - t_pi_start
            if not converge:
                if pi is not None:
                    V_out = p_pi.dot(V)
                    V_out = np.minimum((V_out - max_reward) * self._gamma,
                                       self._reward - max_reward) + max_reward

                    #V_out = np.minimum(V_out, self._reward)
                    return V_out, t_pi
                else:
                    V_mat = np.zeros([self.num_actions2, self.num_actions,
                                      self.num_states])
                    for a2 in range(self.num_actions2):
                        for a1 in range(self.num_actions):
                            p_pi = self.p_trans[a2, a1]
                            V_out = p_pi.dot(V)
                            V_out = np.minimum(
                                (V_out - max_reward) * self._gamma,
                                self._reward - max_reward) + max_reward
                            V_mat[a2, a1, :] = V_out
                    return V_mat, t_pi

            tol = self.tol
            err = tol * 2

            if pi is not None:
                V_pi = deepcopy(V)
                while err > tol:
                    V_old = deepcopy(V_pi)
                    V_pi = p_pi.dot(V_old)
                    V_pi = np.minimum((V_pi - max_reward) * self._gamma,
                                      self._reward - max_reward) + max_reward
                    err = np.linalg.norm(V_old - V_pi, ord=float('inf'))
                return V_pi, t_pi
            else:
                V_mat = np.zeros([self.num_actions2, self.num_actions,
                                      self.num_states])
                for a2 in range(self.num_actions2):
                    V_pi = deepcopy(V)
                    for a1 in range(self.num_actions):
                        while err > tol:
                            V_old = deepcopy(V_pi)
                            p_pi = self.p_trans[a2, a1]
                            V_pi = p_pi.dot(V_old)
                            V_pi = np.minimum((V_pi - max_reward) * self._gamma,
                                              self._reward - max_reward) + max_reward
                            err = np.linalg.norm(V_old - V_pi, ord=float('inf'))
                            print(err)
                    V_mat[a2, a1 :] = V_pi
                return V_mat, t_pi



        else:
            t_pi_start = time.time()
            p_pi = self._p_trans[act2_pi, act_pi, range(nS), :]
            t_pi = time.time() - t_pi_start
            if not converge:
                V_out = p_pi.dot(V)
                V_out = np.minimum((V_out - max_reward) * self._gamma,
                              self._reward - max_reward) + max_reward
                return V_out, t_pi

            tol = self.tol
            err = tol * 2
        
            V_pi= deepcopy(V)
            while err > tol:
                V_old = deepcopy(V_pi)
                V_pi = p_pi.dot(V_old)
                V_pi = np.minimum((V_pi - max_reward) * self._gamma,
                              self._reward - max_reward) + max_reward
                err = np.linalg.norm(V_old - V_pi, ord= float('inf'))
                print(err)
            return V_pi, t_pi

    def _bellman_backup(self, V=None):
        """Performs one bellman backup on the value function V."""

        if V is None:
            V = self._reward
        
        nS = self.num_states

        ones_vec = np.ones([self.num_states, 2]).astype(int)
        pi = ones_vec * 0

        V_mat = np.zeros([self.num_actions2, self.num_actions,
                          self.num_states])

        if not self.sparse:
            for act2 in range(0, self.num_actions2):
                for act in range(0, self.num_actions):
                    pi = ones_vec * np.array([act, act2])
                    V_mat[act2, act, :], _ = self._policy_backup(V, pi)
        else:
            V_mat, _ = self._policy_backup(V)

        
        if self.num_actions == 1 and self.num_actions2 == 1:
            return V_mat[0,0,:], pi
   
        # for act2 in range(0, self.num_actions2)
        #     for act in range(0, self.num_actions):
        #         pi = ones_vec * np.array([act, act2])
        #         V_act = self._policy_backup(V, pi)
        #     pi_greedy = pi_greedy * (V_out >= V_act) +\
        #                 pi * (V_out < V_act)
        #     V_out = np.maximum(V_out, V_act)
        


        temp = np.amin(V_mat, axis=0)
        V_out = np.amax(temp, axis=0)

        pi_greedy = pi
        pi_greedy[:,0] = np.argmax(temp, axis=0)
        pi_greedy[:, 1] = np.argmin(V_mat[:, pi_greedy[:, 0], range(nS)],
                                    axis=0)
        return V_out, pi_greedy

    def _value_iteration(self, V=None, pi=None, one_step=False):
        """Value iteration initialized with initial value function V.
        """
        t_vi_start = time.time()
        if V is None:
            V = self._reward
        V_opt = deepcopy(V)
        tol = self.tol
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
        t_vi = time.time() - t_vi_start
        return V_opt, pi_opt, t_vi, count
    def _policy_iteration(self, V=None, pi=None):
        """Policy iteration initialized with initial value function V."""
        t_pi_start = time.time()
        nS = self.num_states
        if V is None:
            V = self._reward
        
        if pi is None:
            V_pi, pi = self._bellman_backup(V)
        else:
            V_pi = V

        tol = self.tol
        
        print("    err (inf norm)")
        count = 0
        total_overhead = 0
        while True:
            V_old = deepcopy(V_pi)

            V_pi, t_overhead = self._policy_backup(V_old, pi, converge=True)
            total_overhead += t_overhead
            V_opt, pi_opt = self._bellman_backup(V_pi)
            err = np.linalg.norm(V_opt - V_old, ord= float('inf'))
            print("%i:  %.6e" %(count, err))
            count += 1
            # Check for policy convergence.
            if (pi_opt == pi).all() or err<tol:
                break
            pi = pi_opt
            V_pi = V_opt
        t_pi = time.time() - t_pi_start - total_overhead
        return V_opt, pi_opt, t_pi, count

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
        self._v_opt, self._pi_opt, t_run, iter = {'vi': self._value_iteration,
                                     'pi': self._policy_iteration}\
                                     [method](V,pi)
        tot_time = time.time()-t_start
        print("Done. Elapsed time {}.\n".format(tot_time))
        print("Time to run method", t_run)
        return self._v_opt, self._pi_opt
        # return self._v_opt, self._pi_opt, {'tot_time':tot_time,
        #                                    'run_time':t_run, 'iterations':iter}
    
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

    
