import math

import numpy as np
import numpy.linalg as npla

def nearest_le_div(val, d): return val - (val % d)
def nearest_g_div(val, d): return (val + d) - (val % d)


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
 
    s55 = t(0x5555555555555555) & mask  # Add more digits for 128bit support
    s33 = t(0x3333333333333333) & mask
    s0F = t(0x0F0F0F0F0F0F0F0F) & mask
    s01 = t(0x0101010101010101) & mask
    

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))
    
def next_odd(val):
    return int(math.ceil(val)) // 2 * 2 + 1

def rotation_matrix_to_euler(R : np.ndarray) -> np.ndarray:
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def segment_length(p1 : np.ndarray, p2 : np.ndarray):
    """
        p1  (2,)
        p2  (2,)
    """
    return npla.norm(p2-p1)

def segment_to_vector(p1 : np.ndarray, p2 : np.ndarray):
    """
        p1  (2,)
        p2  (2,)
    """
    x = p2-p1
    x /= npla.norm(x)
    return x


def intersect_two_line(a1, a2, b1, b2) -> np.ndarray:
    """
    Returns the point of intersection of the lines (not segments) passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return np.array( [x/z, y/z], np.float32 )

def polygon_area(poly : np.ndarray) -> float:
    """
    calculate area of n-vertices polygon with non intersecting edges

        poly   np.ndarray (n,2)
    """
    return float( np.abs(np.sum( poly[:,0] * np.roll( poly[:,1], -1  ) - poly[:,1] * np.roll( poly[:,0], -1  )  ) / 2) )

