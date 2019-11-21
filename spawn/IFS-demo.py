"""Visualise 2D IFS

Usage:
IFS-demo.py --iter=<n> [--scale=<n>] [--translate=<n>] [--rotate=<n>]
IFS-demo.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import random
import numpy as np
from docopt import docopt
from math import cos,sin
import matplotlib.pyplot as plt
from transformations import rotate


def mk_attractor():
    o = np.array([0.0, 0.0])
    a = np.array([0.8, 0.0])
    b = np.array([0.4, 0.8])
    #c = np.array([0.4, 0.4])
    return np.array([o,a,b])


def plot(X,color,ms,linked=False):
    if linked:
        plt.plot(X[:,0], X[:,1], 'o-', label = 'xy', color=color, ms=ms)
    else:
        plt.plot(X[:,0], X[:,1], 'o', label = 'xy', color=color, ms=ms)


def linear(p,lambd):
    '''Scaling with lambda< 1 decreases all frequencies.'''
    if coinflip():
        scale = 1 + lambd
    else:
        scale = 1 - lambd
    return scale * p

def rotation(p,rho,sign):
    '''Rotations have the effect of increasing/decreasing frequencies -
    they will shift zeros as well as having some effect on collocations.'''
    d1 = 0
    d2 = 1
    rho = rho * sign
    rm = np.identity(2)
    rm[d1][d1] = cos(rho)
    rm[d2][d2] = cos(rho)
    rm[d1][d2] = -sin(rho)
    rm[d2][d1] = sin(rho)
    return np.matmul(p,rm)

def translate(p,att,theta):
    t1 = theta * (p[0] - att[0])
    t2 = theta * (p[1] - att[1])
    #print("P,t1,t2",p,t1,t2)
    p[0] = p[0] - t1
    p[1] = p[1] - t2
    #print("NEWP",p)
    return np.array([p])

def coinflip():
    if random.randint(0,1) == 1:
        return True
    else:
        return False

def get_sign(f):
    if f < 0:
        return -1
    else:
        return 1

if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, IFS 0.1')
    print(args)

    if args["--scale"]:
        lambd = float(args["--scale"])
    if args["--translate"]:
        theta = float(args["--translate"])
    if args["--rotate"]:
        rho = float(args["--rotate"])
    iterations = int(args["--iter"])

    P = mk_attractor()
    plot(P,'red',20,linked=False)

    att = np.random.choice(range(P.shape[0]))
    p = np.array([P[att]])
    points = [p[0]]
    for i in range(iterations):
        if args["--rotate"] and coinflip():
            att = np.random.choice(range(P.shape[0]))
            sign = get_sign(p[0][0]-P[att][0]) 
            pt = rotation(p,rho,sign)
            if pt[0][0] > 0 and pt[0][1] > 0: 
                p = pt
            else:
                p = rotation(p,rho,-sign)
            points.append(p[0])
        if args["--translate"] and coinflip():
            att = np.random.choice(range(P.shape[0]))
            p = translate(p[0],P[att],theta)
            points.append(p[0])
        if args["--scale"] and coinflip():
            p = linear(p,lambd)
            points.append(p[0])
        if coinflip():
            if len(points) != 0:
                plot(np.array(points),'blue',2)
                points.clear()
            att = np.random.choice(range(P.shape[0]))
            p = np.array([P[att]])
            points.append(p[0])
    plot(np.array(points),'blue',2)
    plt.show()
