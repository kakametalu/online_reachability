'''
This file is a function approximator for V for a given theta
It defines a class where we can estimate
'''

import numpy as np

class approximator(object):
    def __init__(self, **kwargs):
        if 'thetas' in kwargs:
            assert 'Vs' in kwargs
            self.theta_vs = {kwargs['thetas'][i]: kwargs['Vs'][i] \
                             for i in range(len(kwargs['thetas']))}
            self.input_shape = len(kwargs['thetas'][0])
            self.output_shape = kwargs['thetas'].shape
        elif 'theta_Vs' in kwargs:
            self.theta_Vs = kwargs['theta_Vs']
            self.input_shape = len(kwargs['thetas_Vs'].keys()[0])
            self.output_shape = kwargs['thetas_Vs'].values()[0].shape
        else:
            self.theta_Vs = {}
            self.input_shape = kwargs['input_shape'] \
                if 'input_shape' in kwargs else None
            self.output_shape = kwargs['output_shape'] \
                if 'output_shape' in kwargs else None

    def insert_new_element(self, theta=None, V=None, theta_V=None):
        if theta is None:
            assert theta_V is not None
            theta, V = theta_V
        else:
            assert V is not None

        if not isinstance(theta, tuple):
            theta = tuple(theta)

        if self.input_shape is None:
            self.input_shape  = len(theta)
        if self.output_shape is None:
            self.output_shape = np.array(V).shape
        assert np.array(V).shape == self.output_shape
        self.theta_Vs[theta] = np.array(V)
        self.theta_Vs[theta] = np.array(V)

    def estimate_V(self, theta):
        return None