# Contains different dynamics models.

import numpy as np
from copy import deepcopy

class Dynamics(object):
    def __init__(self, x_dot, dims, angular=None):
        self._x_dot = x_dot
        self._dims = dims
        self._angular = angular

    def deriv(self, state, control, disturbance):
        """Returns state derivative.

        Args:
            state (2d np array): States to be evaluated.
                Each row corresponds to a state.
            control (1d or 2d np array): Controls to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states.
            disturbance (1d or 2d np array): Disturbance to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states.        
        """
        
        assert state.shape[1] == self._dims,\
          "State dimension is incompatible."
        
        dis = disturbance
        if len(dis.shape) == 1:
            dis_array = np.ones([state.shape[0], dis.shape[0]]) * dis
        else:
            dis_array = dis
        
        if len(control.shape) == 1:
            con_array = np.ones([state.shape[0], control.shape[0]]) * control
        else:
            con_array = control
  
        assert state.shape[0] == con_array.shape[0],\
            "Number of states not equal to number of controls."        
        
        assert state.shape[0] == dis_array.shape[0],\
            "Number of states not equal to number of disturbances."

        return self._x_dot(state, con_array, dis_array)


    def integrate(self, t, state, control, disturbance, steps=10):
        """Intergrate ODE using Runge-Kutta 4 scheme.

        Args:
            t (float): Time interval for integration.
            state (2d np array): States to be evaluated.
                Each row corresponds to a state.
            control (1d or 2d np array): Controls to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states.
            disturbance (1d or 2d np array): Disturbance to be evaluated.
                If 2d, then each row corresponds to a control. There should be a control for each state. If control is 1D then the same control is used for all states.
        """
        dt = t/steps
        run_t = t/steps
        n_state = deepcopy(state) # next states
    	
        #return n_states + t * self.deriv(n_states, control)
        while  run_t <= t:
            k_1 = self.deriv(n_state, control, disturbance)
            k_2 = self.deriv(n_state + dt / 2 * k_1, control, disturbance)
            k_3 = self.deriv(n_state + dt / 2 * k_2, control, disturbance)
            k_4 = self.deriv(n_state + dt * k_3, control, disturbance)
            n_state = n_state + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            run_t += dt

        if self._angular is not None:
            angular = self._angular
            n_state[:, angular] = n_state[:, angular] % (2 * np.pi)
        return n_state


def double_integrator(x, u, d=0, **sys_params):
    """Double integrator Dynamics.
    
    Args:
        x(np array): Current state of the system.
            x[0] - position, angle, velocity, etc.
            x[1] - first derivative.
        u(np array): Control being applied to the system.
        sys_params (dict): Parameters of the system, see below.

    sys_params:
        min_u (float): Minimum applied thrust.
        max_u (float): Maximum applied thrust.
    """
    max_u = sys_params.get('max_u', 1)
    min_u = sys_params.get('min_u', 0)
    output = np.zeros(x.shape)
    output[:, 0] = x[:,1]
    output[:, 1] = u[:,0]* (max_u - min_u) + min_u
    return output


def double_integrator_dist(x, u, d=0, **sys_params):
    """Double integrator Dynamics.
    
    Args:
        x(np array): Current state of the system.
            x[0] - position, angle, velocity, etc.
            x[1] - first derivative.
        u(np array): Control being applied to the system.
        sys_params (dict): Parameters of the system, see below.

    sys_params:
        min_u (float): Minimum applied thrust.
        max_u (float): Maximum applied thrust.
    """
    max_u = sys_params.get('max_u', 1)
    min_u = sys_params.get('min_u', 0)
    max_d = sys_params.get('max_d', 1)
    min_d = sys_params.get('min_d', 0)
    output = np.zeros(x.shape)
    output[:, 0] = x[:,1]
    output[:, 1] = u[:,0] * (max_u - min_u) + min_u +\
        d[:,0] * (max_d - min_d) + min_d
    return output

def dubins_car(x, u, d=0, **sys_params):
    """Dubin's car dynamics.
    
    Args:
        x(np array): Current state of the system.
            x[0] - x position.
            x[1] - y position.
            x[2] - car angle.
        u(np array): Control being applied to the system.
            u[0] - normalized speed.
            u[1] - steering angle.
        sys_params (dict): Parameters of the system, see below.

    sys_params:
        L (float): Axle length.
        max_speed (float): Maximum speed.
    """
    L = sys_params.get('L', 1)
    ms = sys_params.get('max_speed', 1)
    
    output = np.zeros(x.shape)
    output[:, 0] = ms * u[:, 0] * np.cos(x[:, 2])
    output[:, 1] = ms * u[:, 0] * np.sin(x[:, 2])
    output[:, 2] = ms * u[:, 0] / L * np.tan(u[:, 1])

    return output

def pursuit_evasion(x, u, d, **sys_params):
    """Pursuit Evasion Game Dynamics.

        x(np array): Current state of the system.
            x[0] - relative x position.
            x[1] - relative y position.
            x[2] - relative angle.
        u(np array): Control being applied to the system.
            u[0] - angular speed for evader.
        d(np array): Disturbance being applied to the system.
            d[0] - angular speed for pursuer.
        sys_params (dict): Parameters of the system, see below.

        sys_params:
            v_u: Velocity of evader.
            v_d: Velocity of pursuer.
    """

    v_u = sys_params.get('v_u', 5)
    v_d = sys_params.get('v_d', 5)
    output = np.zeros(x.shape)
    output[:, 0] = -v_u + v_d * np.cos(x[:,2]) + u[:,0] * x[:,1]
    output[:, 1] = v_d * np.sin(x[:,2]) - u[:,0] * x[:,0]
    output[:, 2] = u[:,0]-d[:,0]

    return output



# def simple_dyn(x, u, **sys_params):
#     return np.array(u * (1 - np.abs(x)))

# def vdp_oscillator(x, u, **sys_params):
#     """Van Der Pol Oscillator Dynamics."""
#     return np.array([x[1], (1-x[0]**2) * x[1] - x[0] + u[0]])
