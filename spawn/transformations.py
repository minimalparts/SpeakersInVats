import numpy as np
from math import cos,sin

def center(m):
    mean = m.mean(axis=0)
    return m - mean

def scale(m,v):
    #m = center(m)
    dim = m.shape[1]
    sm = np.identity(dim) * v
    return np.matmul(m,sm)

def rotate(m,d1,d2,rho):
    '''Rotate matrix in a particular plane identified by d1,d2.'''
    #m = center(m)
    dim = m.shape[1]
    rm = np.identity(dim)
    rm[d1][d1] = cos(rho)
    rm[d2][d2] = cos(rho)
    rm[d1][d2] = -sin(rho)
    rm[d2][d1] = sin(rho)
    return np.matmul(m,rm)
   
def sum_outer_product(m1, m2):
    return np.einsum('ij,ik->jk', m2, m1)


def find_svd_rotation(m1,m2):
    '''Find best rotation to transform m1 into m2.'''
    m1 = center(m1)
    m2 = center(m2)
    op = sum_outer_product(m2, m1)
    U, S, VT = np.linalg.svd(op)
    rm = U.dot(VT)
    #print("R max cell:",np.max(rm),"R min cell:",np.min(rm))
    return m1.dot(rm)

