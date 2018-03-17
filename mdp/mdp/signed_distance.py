# Module contains signed distance functions

import numpy as np

def hypercube_int(states, cube_lims):
    """Signed distance to a hypercube.
    
    Positive in the interior of the cube, and negative outside.

    Args:
        states (2d np array): States to be evaluated.
        cube_lims (2d np array): Limits of the hypercube.
            First column contains lower limits for each dimension,
            and second column contains upper limits for each dimension.
    Returns:
        dist (1d np array): Signed dist evaluated at states.
    """
    mins = cube_lims[0, :]
    maxs = cube_lims[1, :]
    dist = np.minimum(states - mins, maxs - states)
    dist = np.min(dist, axis=1)
    return dist

def hypersphere_ext(states, center, radius, dims=None):
    """Signed distance to a hypersphere.
    
    Positive in the extrior of the cube, and positive outside.

    Args:
        states (2d np array): States to be evaluated.
        center (1d np array): Center of the hypersphere.
        radius (float): Radius of the hypersphere.
        dims (1d np array): The dims associated with center.
            The distance only varies along these dims. 
    Returns:
        dist (1d np array): Signed dist evaluated at states.
    """

    if dims is None:
        sub_states = states
    else:
        sub_states = states[:, dims]

    dist = np.sum((sub_states - center)**2, axis=1) - radius ** 2

    return dist

def union(shapes):
    """A union of signed distance functions.

    Union is acheived by taking the minimum of signed distances.

    Args:
        shapes (list of funcs): Signed distance functions.
            These are the functions/shapes that will be unioned.
    Returns:
        func (func): Minimum of functions in shapes.
    """

    def func(states):
        output = shapes[0](states)
        for k in range(1,len(shapes)):
            output = np.minimum(output, shapes[k](states))
        return output
    return func

def intersect(shapes):
    """An intersection of signed distance functions.

    Intersection is acheived by taking the maximum of signed distances.

    Args:
        shapes (list of funcs): Signed distance functions.
            These are the functions/shapes that will be instersected.
    Returns:
        func (func): Maximum of functions in shapes.
    """
    
    def func(states):
        output = shapes[0](states)
        for k in range(1,len(shapes)):
            output = np.maximum(output, shapes[k](states))
        return output
    return func

# This function does set(A) - set (B)
# If ext is true, the outside is considered to be positive, apply intersect
# then or apply union
# By default all are -ve inside and +ve outside
def set_minus(set_A, set_B, ext=True):
    return lambda states: np.maximum(set_A(states), -set_B(states)) if ext \
        else np.minimum(-set_A(states), set_B(states))









