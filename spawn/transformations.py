import numpy as np
from math import cos,sin

def center(m):
    mean = m.mean(axis=0)
    return m - mean

def stretch(m,v):
    m = center(m)
    dim = m.shape[1]
    sm = np.identity(dim) * v
    return np.matmul(m,sm)

def rotate(m,d1,d2,theta):
    '''Rotate matrix in a particular plane identified by d1,d2.'''
    m = center(m)
    dim = m.shape[1]
    rm = np.identity(dim)
    rm[d1][d1] = cos(theta)
    rm[d2][d2] = cos(theta)
    rm[d1][d2] = sin(theta)
    rm[d2][d1] = -sin(theta)
    return np.matmul(m,rm)
    
    
