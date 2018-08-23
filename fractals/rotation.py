import numpy as np
import matplotlib.pyplot as plt
from random import randint
from math import cos, sin
from scipy.spatial import ConvexHull

def convexhull(p):
    hull = ConvexHull(p)
    return hull

def rotate_coords(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).

    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy

def rotate(A, centroid):
    rotation = np.array([(cos(0.1), -sin(0.1)),(sin(0.1),cos(0.1))])
    return np.dot(A,rotation.T) + centroid

curr_point = [0,0]  # our seed value for the chaos game
                    # It can fall anywhere inside the triangle

# our equilateral triangle vertices
v1 = [0.3,0.4]
v2 = [0.3,0.2]
v3 = [0.2,0.5]
A = np.array([v1,v2,v3])
centroid = np.mean(A)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Plot 5000 points
for _ in range(20):
    # choose a triangle vertex at random
    # set the current point to be the midpoint
    # between the previous current point and
    # the randomly chosen vertex
    hull = convexhull(A)
    plt.scatter(A[:,0],A[:,1])
    #plt.plot(poly[:,0], poly[:,1])
    for simplex in hull.simplices:
        plt.plot(A[simplex,0], A[simplex,1], 'k-')
    R = rotate(A-centroid, centroid) 
    A = R
    centroid = np.mean(A)

plt.show()

