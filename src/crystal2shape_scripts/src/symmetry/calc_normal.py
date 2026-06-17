import numpy as np
    
    
def calc_normal_func(points):
    """
    Calculate the normal vector of a plane defined by three points.

    Parameters
    ----------
    points : list of list or ndarray
        A list containing three points in 3D space, each represented as a list or an array of three coordinates.

    Returns
    -------
    n_unit : ndarray
        A unit normal vector to the plane defined by the three points.
    """
    
    a = np.array(points[1]) - np.array(points[0])
    b = np.array(points[2]) - np.array(points[0])
    n_cap = np.cross(a, b)
    n_unit = (n_cap/np.linalg.norm(n_cap))
    d = -n_unit[0]*points[0][0] - n_unit[1]*points[0][1] - n_unit[2]*points[0][2]

    return n_unit[:]