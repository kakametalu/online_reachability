# Contains different dynamics models.

import numpy as np
from copy import deepcopy

class Dynamics(object):
    def __init__(self, x_dot, dims):
        self._x_dot = x_dot
        self._dims = dims

    def deriv(self, state, control):
        """Returns state derivative.

        Args:
            state (2d np array): States to be evaluated.
                Each row corresponds to a state.
            control (1d or 2d np array): Controls to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states. 
        """
        
        assert state.shape[1] == self._dims,\
          "State dimension is incompatible."
        
        if len(control.shape) == 1:
            return np.array([self._x_dot(state, control) for state in state])
        else:
            assert state.shape[0] == control.shape[0],\
             "Number of states not equal to number of controls."
            return np.array([self._x_dot(x, u)
                             for x, u in zip(state, control)])

    def integrate(self, state, control, t, steps=10):
        """Intergrate ODE using Runge-Kutta 4 scheme.

        Args:
            state (2d np array): States to be evaluated.
                Each row corresponds to a state.
            control (1d or 2d np array): Controls to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states. 
        """
        dt = t/steps
        run_t = t/steps
        n_state = deepcopy(state) # next states
    	
        #return n_states + t * self.deriv(n_states, control)
        while  run_t <= t:
            k_1 = self.deriv(n_state, control)
            k_2 = self.deriv(n_state + dt / 2 * k_1, control)
            k_3 = self.deriv(n_state + dt / 2 * k_2, control)
            k_4 = self.deriv(n_state + dt * k_3, control)
            n_state = n_state + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            run_t += dt
        return n_state


def double_integrator(state, control, **sys_params):
    """Double integrator Dynamics.
    
    Args:
        state(np array): Current state of the system.
        control(np array): Control being applied to the system.
        sys_params (dict): Parameters of the system, see below.

    sys_params:
        min_u (float): Minimum applied thrust.
        max_u (float): Maximum applied thrust.
    """
    max_u = sys_params.get('max_u', 1)
    min_u = sys_params.get('min_u', 0)
    return np.array([state[1], control * (max_u - min_u) + min_u])

def simple_dyn(x, u, **sys_params):
    return np.array(u * (1 - np.abs(x)))

def vdp_oscillator(x, u, **sys_params):
    """Van Der Pol Oscillator Dynamics."""
    return np.array([x[1], (1-x[0]**2) * x[1] - x[0] + u[0]])
