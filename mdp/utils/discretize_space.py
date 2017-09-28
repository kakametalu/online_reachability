'''
In this file, we have helper functions for discretizing spaces
'''

import numpy as np
import itertools

# Dicretize based on epsilon
# This returns an iterator instead of as sometimes list might be too big
def discretize_epsilon(bounds, epsilon):
    '''First assert that either have a single epsilon or as many epsilons as
       there are dimensions in bounds'''
    assert not hasattr(epsilon, "__len__") or len(epsilon)==len(bounds)
    if not hasattr(epsilon, "__len__"):
        epsilon = np.repeat(epsilon, len(bounds))

    bound_arrays = [np.arange(bound[0], bound[1]+eps, eps) for bound, eps in \
                    zip(bounds, epsilon)]

    return itertools.product(*bound_arrays)

# Discretize based on N
# This returns an iterator instead of as sometimes list might be too big
def discretize_N(bounds, N, each_dim=True):
    '''First assert that either have a single N or as many Ns as there are
       dimensions in bounds'''
    assert not hasattr(N, "__len__") or len(N) == len(bounds)
    if not hasattr(N, "__len__"):
        if each_dim:
            N = np.repeat(N, len(bounds))
        else:
            N = np.ceil(N/len(bounds)).astype(int)
            N = np.repeat(N, len(bounds))

    bound_arrays = [np.linspace(bound[0], bound[1], n) for bound, n in\
                    zip(bounds, N)]

    return itertools.product(*bound_arrays)
