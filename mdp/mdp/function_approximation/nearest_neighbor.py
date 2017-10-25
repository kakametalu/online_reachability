'''
The estimated V is the V of the theta closest to the querying theta
'''

import numpy as np
from .func_approx import approximator

class nearest_neighbor(approximator):
    def __init__(self, **kwargs):
        super(nearest_neighbor, self).__init__(**kwargs)

    def estimate_V(self, theta):
        set_of_thetas = np.array(list(self.theta_Vs.keys()))
        nn_id = np.abs(set_of_thetas-theta).argmin()
        nn_theta = set_of_thetas[nn_id]
        if self.input_shape > 1:
            nn_theta = tuple(nn_theta)
        return self.theta_Vs[nn_theta]



