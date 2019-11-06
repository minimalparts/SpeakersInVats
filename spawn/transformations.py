import numpy as np

def center(m):
    mean = m.mean(axis=0)
    return m - mean

def stretch(m,v):
    m = center(m)
    dim = m.shape[1]
    sm = np.identity(dim) * v
    return np.matmul(m,sm)

def rotate(m,theta):
    m = center(m)
    dim = m.shape[1]
    
