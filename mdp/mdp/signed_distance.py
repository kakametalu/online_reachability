# Module contains signed distance functions

import numpy as np

def dist_hypercube_int(states, cube_lims):
    """Signed distance to a hypercube.
    
    Positive in the interior of the cube, and negative outside.

    Args:
        states (2d np array): States to be evaluated.
        cube_lims (2d np array): Limits of the hypercube.
            First column contains lower limits for each dimension,
            and second column contains upper limits for each dimension.
    """
    mins = cube_lims[0, :]
    maxs = cube_lims[1, :]
    dist = np.minimum(states - mins, maxs - states)
    dist = np.min(dist, axis=1)
    return dist
