import numpy as np

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

    def add_transition(self, state, action, next_states, probs):
        """Add new transition.

        Args:
            state (uint): State index.
            action (uint): Action index.
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
        """
        assert (state < self._num_states),\
            "State index {} out of range.".format(state)
        assert (action < self._num_actions), "Action index out of range."
        assert (np.sum(probs) == 1.0), "Total prob = {}".format(np.sum(probs))
        for next_state in next_states:
            assert (next_state < self._num_states),\
                "Next state {} index out of range.".format(next_state)

        self._trans_dict.update({(state, action): (next_states, probs)})
        self._trans_tensor[action, state, next_states] = probs

    def __getitem__(self, state, action):
        """Takes state and action and returns next states and probabilities.

        Args:
            state (uint): State index.
            action (uint): Action index.

        Returns:
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
        """

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

    Attributes:
        _reward_func (func): Scalar reward function.
            Usage reward = _reward_func(state, action, next_state).
        _gamma (float): Discount rate between 0 and 1.
        _expR (np array): Expected immediate reward for state action pair.
            Shape is num_states by num_actions.


    Args:
        num_states (uint): Number of states.
        num_actions(uint): Number of actions.
        _reward_func (func): Scalar reward function.
            Usage reward = _reward_func(state, action, next_state).
        gamma (float): Discount rate between 0 and 1.
    """

    def __init__(self, num_states, num_actions, reward_func = None,
                 gamma=None):
        """Initialize MDP Object."""
        
        if reward_func is None:
            reward_func = zero_reward

        if gamma is None:
            gamma = 0.95
                
        super().__init__(num_states, num_actions)
        self._reward_func = reward_func
        self._gamma = gamma
        self._expR = np.zeros([num_states, num_actions])


    def add_transition(self, state, action, next_states, probs):
        """Add new transition and expected reward.

        Args:
            state (uint): State index.
            action (uint): Action index.
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
        """
        super().add_transition(state, action, next_states, probs)
        self._expR[state, action] = np.sum([self.reward(state, action, x[0])
                                            * x[1]for x in zip(next_states,
                                                               probs)])
    def __getitem__(self, state, action):
        """Takes state and action and returns (next_states, probs, exp rewards).

        Args:
            state (uint): State index.
            action (uint): Action index.

        Returns:
            next_states (1D np array): Possible next transition stats.
            probs (1D np array): Transition probabilities.
            rewards (float): Reward for state/action pair.
        """
        next_states, probs = super().__getitem__(state, action)
        exp_reward = self._expR[state, action]
        return next_states, probs, exp_reward

    def reward(self, state, action, next_state):
        """Return reward."""

        return self._reward_func(state, action, next_state)

    def _policy_back_up(self, V, pi):
        """Does one policy back up on the value function."""

        V_out = np.zeros(self._num_states)
        for state in range(self._num_states):
            next_support, ps, exp_r= self[state, pi[state]]
            V_out[state] = exp_r
            for next, p in zip(next_support, ps):
                V_out[state] += V[next] * p
        return V_out

    def _bellman_backup(self, V):
        """Performs one bellman backup on the value function."""
        pi = np.ones(super().num_actions).astype(int) * 0
        V_out = self._policy_back_up(self,V, pi)

        for act in range(1, super().num_actions):
            pi = np.ones(super().num_actions).astype(int) * act
            V_act = self._policy_back_up(self, V, pi)
            V_out = np.maximum(V_out, V_temp)

        return V_out



# def inf_norm(v):
#     "Returns infinity norm of vecotr v"
#     return np.max(np.abs(v))


# def lsr_model(thetas, nS):
#     """ Simple transition model for moving left, right, or staying

#     The states are arranged along a line. Each action has a probibility for
#     moving to the left, right, or stayin in place. State 0 and state nS-1 are
#     pas states and state nS is an absorbing state.

#     Parameters
#     ----------
#     move : :
#         theta[a]= [theta_a_l,theta_a_s,theta_a_r]: transition probability of left, stay, right for action a
#             P[a,s,s']=
#                 theta_a_l for s'= s-1, s non-PAS
#                 theta_a_s for s'= s, s non-PAS
#                 theta_a_r for s'= s+1, s non-PAS
#     Output:
#         P - transition model
#         P_grad - model gradient
#     """
#     nA = len(thetas)
#     P = []

#     P_grad = []

#     int_pts = range(1, nS - 2)
#     right_pts = [s + 1 for s in int_pts]
#     left_pts = [s - 1 for s in int_pts]

#     for a in range(nA):
#         assert (thetas[a] >= 0.0).all()
#         theta_normalizer = sum(thetas[a])

#         P_a = np.zeros([nS, nS])  # transition matrix for action a

#         P_grad_a = np.zeros([3, nS, nS])  # gradient tensor for action a

#         # transition probability matrix
#         P_a[int_pts, left_pts] = thetas[a][0] / theta_normalizer  # left
#         P_a[int_pts, int_pts] = thetas[a][1] / theta_normalizer  # stay
#         P_a[int_pts, right_pts] = thetas[a][2] / theta_normalizer  # right
#         P_a[0, nS - 1] = 1.0
#         P_a[nS - 2, nS - 1] = 1.0
#         P_a[nS - 1, nS - 1] = 1.0

#         # gradient w.r.t theta_l
#         P_grad_a[0][int_pts, left_pts] = (theta_normalizer - thetas[a][
#             0]) / theta_normalizer ** 2
#         P_grad_a[0][int_pts, int_pts] = -thetas[a][1] / theta_normalizer ** 2
#         P_grad_a[0][int_pts, right_pts] = -thetas[a][2] / theta_normalizer ** 2

#         # gradient w.r.t theta_s
#         P_grad_a[1][int_pts, left_pts] = -thetas[a][0] / theta_normalizer ** 2
#         P_grad_a[1][int_pts, int_pts] = (theta_normalizer - thetas[a][
#             1]) / theta_normalizer ** 2
#         P_grad_a[1][int_pts, right_pts] = -thetas[a][2] / theta_normalizer ** 2

#         # gradient w.r.t theta_r
#         P_grad_a[2][int_pts, left_pts] = -thetas[a][0] / theta_normalizer ** 2
#         P_grad_a[2][int_pts, int_pts] = -thetas[a][1] / theta_normalizer ** 2
#         P_grad_a[2][int_pts, right_pts] = (theta_normalizer - thetas[a][
#             2]) / theta_normalizer ** 2

#         P_grad.append(P_grad_a)
#         P.append(P_a)

#     return np.array(P), np.swapaxes(np.array(P_grad), 0, 1)


# def keep_out_indicator(keep_out, nS, nA):
#     """
#     reward function indicating when keepout set transitions to absorbing state.
#     R[a,s,s']=
#         1, s in keep_out s' absorbing state
#         0, o/w
#     Args:
#         keep_out: list of indices of states in keep-out set
#         ns - number of states
#         nA - number of actions
#     Returns:
#         R - reward model
#     """
#     R = np.zeros([nA, nS, nS])
#     for a in range(nA):
#         R[a, keep_out, -1] = 1.0
#     return R


# def bellman(P_pi, exp_reward, V, gamma=1.0):
#     """
#     Applies bellman operator to value function

#     Args:
#         V - value function, 1D np array (size nS)
#         exp_reward - exp_reward, 1D array (size nS)
#         P_pi - transition matrix (under a fixed policy, pi)
#     """

#     V_out = exp_reward + gamma * np.dot(P_pi, V)
#     return V_out


# def optimal_policy(P, R, V, gamma=1.0):
#     # Returns the greedy policy given a value function, transition model, and reward model

#     nA, nS, _ = P.shape
#     V_hold = np.ones(nS) * 1000
#     opt_pi = [0] * nS

#     for a in range(nA):
#         pi = [a] * nS
#         P_pi = P[pi, range(nS), :]
#         exp_reward = np.sum(R[pi, range(nS), :] * P_pi, axis=1)
#         temp = bellman(P_pi, exp_reward, V, gamma)
#         opt_pi = [a if i < j else k for (i, j, k) in zip(temp, V_hold, opt_pi)]
#         V_hold = np.minimum(V_hold, temp)
#     return opt_pi


# def policy_evaluation(P, R, V=None, pi=None, gamma=1.0, solve_lin=False):
#     # computes the value function for a given policy, transition model, and reward model

#     nS = P.shape[-1]
#     V_pi = V

#     if V_pi is None:
#         V_pi = np.zeros(nS)
#     if pi is None:
#         pi = [0] * nS

#     P_pi = P[pi, range(nS), :]

#     exp_reward = np.sum(R[pi, range(nS), :] * P_pi, axis=1)

#     if solve_lin is True:
#         V = np.zeros(nS)
#         V[:nS - 1] = np.linalg.solve(np.eye(nS - 1) - P_pi[:nS - 1, :nS - 1],
#                                      exp_reward[:nS - 1])
#         # print('condition number: ', np.linalg.cond(np.eye(nS-1)-P_pi[:nS-1,:nS-1]))

#     count = 0
#     tol = 10 ** (-8)
#     while True:
#         V_old = V_pi
#         V_pi = exp_reward + gamma * np.dot(P_pi, V_old)
#         err_inf = np.max(np.abs(V_pi - V_old) / (np.abs(V_old) + tol))
#         # print(err_inf)
#         count += 1

#         if err_inf < tol:
#             break

#     return V_pi


# def policy_iteration(P, R, V=None, pi=None, gamma=1.0, solve_lin=False):
#     # finds the optimal policy and value function
#     # solve_lin: 0 (default) - solve iteratively, 1 - solve a linear system (matrix inversion)

#     now_pi = pi
#     V_pi = V

#     # initialization
#     if V_pi is None:
#         V_pi = policy_evaluation(P, R, V_pi, now_pi, gamma, solve_lin)
#     now_pi = optimal_policy(P, R, V_pi, gamma)

#     tol = 10 ** (-8)
#     while True:
#         old_pi = now_pi
#         V_old = V_pi
#         V_pi = policy_evaluation(P, R, V_pi, now_pi, gamma, solve_lin)
#         now_pi = optimal_policy(P, R, V_pi, gamma)

#         # print(now_pi[0:100])
#         err_inf = np.max(np.abs(V_pi - V_old) / (np.abs(V_old) + tol))
#         # print(err_inf)

#         if err_inf < tol:
#             break

#     return V_pi, now_pi


# class Environment(object):
#     def __init__(self, P, R):
#         self.P = P
#         self.R = R

#     def simulate(self, state, action):
#         select = np.random.rand(1)

#         next_state_cdf = np.cumsum(self.P[action, state, :])

#         idx = 0
#         while True:
#             if select < next_state_cdf[idx]:
#                 break
#             idx += 1
#         next_state = idx
#         reward = self.R[action, state, next_state]

#         return next_state, reward


# def td_n_step(V_old, reward_trace, state_trace, gamma=1.0, alpha=0.1, N=None):
#     if N is None:
#         N = len(reward_trace)
#     N = min(len(reward_trace), N)

#     V_next = V_old + 0.0
#     reward = 0
#     nS = V_old.size
#     terminal = 0

#     for k in range(N):
#         reward += gamma ** k * reward_trace[k]
#         if state_trace[k] is nS - 1:
#             terminal = 1
#             break

#     V_next[state_trace[0]] = V_next[state_trace[0]] * (1 - alpha) + (reward + (
#     1 - terminal) * gamma ** N * V_next[state_trace[N]]) * (alpha)
#     return V_next


# def td_lambda(V_old, reward, state, state_next, e_trace=None, lamb=0, gamma=1,
#               alpha=0.1):
#     if e_trace is None:
#         e_trace = np.zeros(V_old.size)

#     e_trace_next = e_trace * gamma * lamb
#     e_trace_next[state] += 1
#     err = reward + V_old[state_next] - V_old[state]
#     grad = err * alpha * e_trace_next
#     V_next = V_old + grad
#     return V_next, e_trace_next


# """
# def td_lambda_adadelta(V_old,reward,state,state_next,acc_grad,acc_delta,e_trace=None,lamb=0,gamma=1,rho=0.5,alpha=0.1):
#     eps=10**(0)
#     if e_trace is None:
#         e_trace= np.zeros(V_old.size)

#     e_trace_next=e_trace*gamma*lamb
#     e_trace_next[state]+=1
#     err=reward+V_old[state_next]-V_old[state]
#     grad=err*e_trace_next
#     acc_grad = acc_grad*rho+(1-rho)*grad**2
#     delta = 1/(acc_grad+eps)**(0.5)*grad*alpha
#     acc_delta = acc_delta*rho+(1-rho)*delta**2
#     V_next=V_old+delta
#     return V_next, e_trace_next, acc_grad, acc_delta


# def td_lambda_adadelta_approx(V_old,reward,state,state_next,acc_grad,acc_delta,e_trace=None,lamb=0,gamma=1,rho=0.5,alpha=0.1):
#     eps=10**(0)
#     if e_trace is None:
#         e_trace= np.zeros(V_old.size)

#     e_trace_next=e_trace*gamma*lamb
#     e_trace_next[state]+=1
#     err=reward+V_old[state_next]-V_old[state]
#     grad=err*e_trace_next
#     acc_grad = acc_grad*rho+(1-rho)*grad**2
#     delta = 1/(acc_grad+eps)**(0.5)*grad*alpha
#     acc_delta = acc_delta*rho+(1-rho)*delta**2
#     V_next=V_old+delta
#     return V_next, e_trace_next, acc_grad, acc_delta
# """

# import numpy as np
# from scipy import sparse
# import time

# """
# This module contains a host of functions for online learning of the value
# function for the reachabiulity problem. This module will inculde different MDP
# environments, and different learning algorithms. 

# We consider MDPs over discrete states and use the following convention:

#     nS - number of states including absorbing state, absorbing state is last state
#     pas-pre-absorbing states, state that must transition to absorbing state with probability one
#     nA - number of actions
#     P - transition probability model, np 3D array P[a,s,s'], a-current action, s'-next state, s-current state
#     R - reward model, np 3D array R[a,s,s'], a-current action, s'-next state, s-current state
#     P_grad - gradient of the transition probability model np 4D array P_grad[p,a,s,s'], p-parameter, a-action, s'- next state, s-current state

# All MDP model generator functions end with _model in the name and return a 
# transition probability P
# """

# def inf_norm(v):
#     "Returns infinity norm of vecotr v"
#     return np.max(np.abs(v))

# def lsr_model(thetas,nS):
#     """ Simple transition model for moving left, right, or staying

#     The states are arranged along a line. Each action has a probibility for
#     moving to the left, right, or stayin in place. State 0 and state nS-1 are
#     pas states and state nS is an absorbing state.

#     Parameters
#     ----------
#     move : :
#         theta[a]= [theta_a_l,theta_a_s,theta_a_r]: transition probability of left, stay, right for action a
#             P[a,s,s']=
#                 theta_a_l for s'= s-1, s non-PAS
#                 theta_a_s for s'= s, s non-PAS
#                 theta_a_r for s'= s+1, s non-PAS
#     Output:
#         P - transition model
#         P_grad - model gradient
#     """
#     nA = len(thetas)
#     P=[]

#     P_grad=[]

#     int_pts=range(1,nS-2)
#     right_pts= [s+1 for s in int_pts]
#     left_pts= [s-1 for s in int_pts]

#     for a in range(nA):
#         assert (thetas[a]>=0.0).all()
#         theta_normalizer=sum(thetas[a])

#         P_a=np.zeros([nS,nS])  #transition matrix for action a

#         P_grad_a=np.zeros([3,nS,nS]) #gradient tensor for action a

#         #transition probability matrix
#         P_a[int_pts,left_pts]=thetas[a][0]/theta_normalizer #left
#         P_a[int_pts,int_pts]=thetas[a][1]/theta_normalizer #stay
#         P_a[int_pts,right_pts]=thetas[a][2]/theta_normalizer #right
#         P_a[0,nS-1]=1.0
#         P_a[nS-2,nS-1]=1.0
#         P_a[nS-1,nS-1]=1.0

#         #gradient w.r.t theta_l
#         P_grad_a[0][int_pts,left_pts] = (theta_normalizer-thetas[a][0])/theta_normalizer**2
#         P_grad_a[0][int_pts,int_pts]=-thetas[a][1]/theta_normalizer**2
#         P_grad_a[0][int_pts,right_pts]=-thetas[a][2]/theta_normalizer**2

#         #gradient w.r.t theta_s
#         P_grad_a[1][int_pts,left_pts] =-thetas[a][0]/theta_normalizer**2
#         P_grad_a[1][int_pts,int_pts]=(theta_normalizer-thetas[a][1])/theta_normalizer**2
#         P_grad_a[1][int_pts, right_pts]=-thetas[a][2]/theta_normalizer**2

#         #gradient w.r.t theta_r
#         P_grad_a[2][int_pts,left_pts] =-thetas[a][0]/theta_normalizer**2
#         P_grad_a[2][int_pts,int_pts]=-thetas[a][1]/theta_normalizer**2
#         P_grad_a[2][int_pts,right_pts]=(theta_normalizer-thetas[a][2])/theta_normalizer**2

#         P_grad.append(P_grad_a)
#         P.append(P_a)

#     return np.array(P), np.swapaxes(np.array(P_grad),0,1)

# def keep_out_indicator(keep_out,nS,nA):
#     """
#     reward function indicating when keepout set transitions to absorbing state.
#     R[a,s,s']=
#         1, s in keep_out s' absorbing state
#         0, o/w
#     Args:
#         keep_out: list of indices of states in keep-out set
#         ns - number of states
#         nA - number of actions
#     Returns:
#         R - reward model
#     """
#     R=np.zeros([nA,nS,nS])
#     for a in range(nA):
#         R[a,keep_out,-1]=1.0
#     return R


# def bellman(P_pi, exp_reward, V, gamma=1.0):
#     """
#     Applies bellman operator to value function

#     Args:
#         V - value function, 1D np array (size nS)
#         exp_reward - exp_reward, 1D array (size nS)
#         P_pi - transition matrix (under a fixed policy, pi)
#     """

#     V_out=exp_reward+gamma*np.dot(P_pi,V)
#     return V_out

# def optimal_policy(P,R,V,gamma=1.0):
#     #Returns the greedy policy given a value function, transition model, and reward model

#     nA,nS,_=P.shape
#     V_hold=np.ones(nS)*1000
#     opt_pi=[0]*nS

#     for a in range(nA):
#         pi=[a]*nS
#         P_pi=P[pi,range(nS),:]
#         exp_reward=np.sum(R[pi,range(nS),:]*P_pi,axis=1)
#         temp=bellman(P_pi,exp_reward,V,gamma)
#         opt_pi=[a if i<j else k for (i,j,k) in zip(temp,V_hold,opt_pi)]
#         V_hold=np.minimum(V_hold,temp)
#     return opt_pi

# def policy_evaluation(P, R, V=None, pi=None,gamma=1.0,solve_lin=False):
#     # computes the value function for a given policy, transition model, and reward model

#     nS=P.shape[-1]
#     V_pi=V

#     if V_pi is None:
#         V_pi=np.zeros(nS)
#     if pi is None:
#         pi = [0]*nS

#     P_pi=P[pi,range(nS),:]

#     exp_reward=np.sum(R[pi,range(nS),:]*P_pi,axis=1)

#     if solve_lin is True:
#         V=np.zeros(nS)
#         V[:nS-1]=np.linalg.solve(np.eye(nS-1)-P_pi[:nS-1,:nS-1],exp_reward[:nS-1])
#         #print('condition number: ', np.linalg.cond(np.eye(nS-1)-P_pi[:nS-1,:nS-1]))

#     count=0
#     tol=10**(-8)
#     while True:
#         V_old = V_pi
#         V_pi =exp_reward+gamma*np.dot(P_pi,V_old)
#         err_inf=np.max(np.abs(V_pi-V_old)/(np.abs(V_old)+tol))
#         #print(err_inf)
#         count+=1


#         if err_inf<tol:
#             break


#     return V_pi

# def policy_iteration(P,R,V=None,pi=None, gamma=1.0, solve_lin=False):
#     #finds the optimal policy and value function
#     #solve_lin: 0 (default) - solve iteratively, 1 - solve a linear system (matrix inversion)

#     now_pi=pi
#     V_pi=V

#     #initialization
#     if V_pi is None:
#         V_pi=policy_evaluation(P,R,V_pi,now_pi,gamma,solve_lin)
#     now_pi=optimal_policy(P,R,V_pi,gamma)

#     tol=10**(-8)
#     while True:
#         old_pi=now_pi
#         V_old=V_pi
#         V_pi=policy_evaluation(P,R,V_pi,now_pi,gamma,solve_lin)
#         now_pi=optimal_policy(P,R,V_pi,gamma)

#         #print(now_pi[0:100])
#         err_inf=np.max(np.abs(V_pi-V_old)/(np.abs(V_old)+tol))
#         #print(err_inf)

#         if err_inf<tol:
#             break


#     return V_pi, now_pi

# class Environment(object):
#     def __init__(self,P,R):
#         self.P=P
#         self.R=R
#     def simulate(self,state,action):
#         select=np.random.rand(1)

#         next_state_cdf=np.cumsum(self.P[action,state,:])

#         idx=0
#         while True:
#             if select<next_state_cdf[idx]:
#                 break
#             idx+=1
#         next_state=idx
#         reward=self.R[action,state,next_state]

#         return next_state, reward


# def td_n_step(V_old,reward_trace,state_trace,gamma=1.0,alpha=0.1,N=None):

#     if N is None:
#         N=len(reward_trace)
#     N=min(len(reward_trace),N)

#     V_next=V_old+0.0
#     reward=0
#     nS=V_old.size
#     terminal=0

#     for k in range(N):
#         reward+=gamma**k*reward_trace[k]
#         if state_trace[k] is nS-1:
#             terminal=1
#             break

#     V_next[state_trace[0]]=V_next[state_trace[0]]*(1-alpha)+(reward+(1-terminal)*gamma**N*V_next[state_trace[N]])*(alpha)
#     return V_next

# def td_lambda(V_old,reward,state,state_next,e_trace=None,lamb=0,gamma=1,alpha=0.1):
#     if e_trace is None:
#         e_trace= np.zeros(V_old.size)

#     e_trace_next=e_trace*gamma*lamb
#     e_trace_next[state]+=1
#     err=reward+V_old[state_next]-V_old[state]
#     grad=err*alpha*e_trace_next
#     V_next=V_old+grad
#     return V_next, e_trace_next

# """
# def td_lambda_adadelta(V_old,reward,state,state_next,acc_grad,acc_delta,e_trace=None,lamb=0,gamma=1,rho=0.5,alpha=0.1):
#     eps=10**(0)
#     if e_trace is None:
#         e_trace= np.zeros(V_old.size)
        
#     e_trace_next=e_trace*gamma*lamb
#     e_trace_next[state]+=1
#     err=reward+V_old[state_next]-V_old[state]
#     grad=err*e_trace_next
#     acc_grad = acc_grad*rho+(1-rho)*grad**2
#     delta = 1/(acc_grad+eps)**(0.5)*grad*alpha
#     acc_delta = acc_delta*rho+(1-rho)*delta**2
#     V_next=V_old+delta
#     return V_next, e_trace_next, acc_grad, acc_delta


# def td_lambda_adadelta_approx(V_old,reward,state,state_next,acc_grad,acc_delta,e_trace=None,lamb=0,gamma=1,rho=0.5,alpha=0.1):
#     eps=10**(0)
#     if e_trace is None:
#         e_trace= np.zeros(V_old.size)
        
#     e_trace_next=e_trace*gamma*lamb
#     e_trace_next[state]+=1
#     err=reward+V_old[state_next]-V_old[state]
#     grad=err*e_trace_next
#     acc_grad = acc_grad*rho+(1-rho)*grad**2
#     delta = 1/(acc_grad+eps)**(0.5)*grad*alpha
#     acc_delta = acc_delta*rho+(1-rho)*delta**2
#     V_next=V_old+delta
#     return V_next, e_trace_next, acc_grad, acc_delta
# """