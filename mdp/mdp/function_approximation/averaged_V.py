'''
The estimated theta is a weighted average of V's of thetas closest to the
querying theta
'''

import numpy as np
from .func_approx import approximator


class averaged_V(approximator):
    def __init__(self, **kwargs):
        super(averaged_V, self).__init__(**kwargs)
        # For k nearest neighbor
        # Defaulted to all the thetas that are all seen
        self.k = kwargs['k_nearest'] if 'k_nearest' in kwargs else None


    def estimate_V(self, theta, weighting=None):
        set_of_thetas = np.array(list(self.theta_Vs.keys()))
        if self.k is None or len(set_of_thetas)  < self.k:
            k = len(set_of_thetas)
        else:
            k = self.k

        sum_dist = np.sum(np.abs(set_of_thetas - theta), axis=1)
        nn_ids = np.argpartition(sum_dist,k-1)[0:k]

        nn_theta_dist = sum_dist[nn_ids]
        if weighting is None:
            nn_theta_dist = nn_theta_dist + 0.001
            nn_theta_dist = 1/nn_theta_dist
        else:
            nn_theta_dist = weighting(nn_theta_dist)
        # Normalizing for weighting based on distance
        nn_theta_dist = nn_theta_dist/np.sum(nn_theta_dist)

        if self.input_shape > 1:
            nn_thetas = [tuple(set_of_thetas[i]) for i in nn_ids]


        return np.sum(np.array([self.theta_Vs[nn_thetas[i]]*nn_theta_dist[i]
                                for i in range(len(nn_theta_dist))]), axis=0)



