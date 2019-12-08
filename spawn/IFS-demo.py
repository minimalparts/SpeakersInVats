"""Visualise 2D IFS

Usage:
IFS-demo.py --iter=<n> --sample=<n> --translate=<n>
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
from utils import centroid
from evals import compute_euclidian

def get_cmap(i):
    vals = np.linspace(0,1,i)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.brg(vals))
    return cmap

def mk_attractor():
    o = np.array([0.0, 0.0])
    a = np.array([0.7, 0.0])
    b = np.array([0.5, 0.7])
    m = np.array([0.3, 0.3])
    return np.array([o,a,b,m])

def plot(X,color,ms,linked=False):
    if linked:
        X = np.vstack([X,X[0]])
        plt.plot(X[:,0], X[:,1], 'o-', label = 'xy', color=color, ms=ms)
    else:
        plt.plot(X[:,0], X[:,1], 'o', label = 'xy', color=color, ms=ms)

def translate(p,att,theta):
    t1 = theta * (p[0] - att[0])
    t2 = theta * (p[1] - att[1])
    #print("OLDP",p,"ATT",att,"DIFF",t1,t2)
    p[0] = p[0] - t1
    p[1] = p[1] - t2
    #print("NEWP",p)
    return p

def plot_attractor(P):
    plt.plot(P[:,0], P[:,1], 'o', label = 'xy', color='black', ms=15)


def mk_color_points_dict(colors):
    points = {}
    for i in range(len(colors)):
        points[i] = []
    return points

def plot_color_points(points,colors):
    for c,p in points.items():
        plot(np.array(p),colors[c],3)

def run_attractor(P,ind,theta,iterations,colors):
    points = mk_color_points_dict(colors)
    att_similarities = compute_euclidian(P)

    p = P[ind].copy()
    for i in range(iterations):
        itheta = theta
        #probs = att_similarities[ind] / sum(att_similarities[ind])
        #att = np.random.choice([i for i in range(P.shape[0])],1,p=probs)[0]
        att = random.choice(range(P.shape[0]))
        #itheta = theta * att_similarities[ind][att]
        #print(ind,"-->",att,itheta)
        p = translate(p,P[att],itheta)
        points[att].append(p.copy())

    plot_color_points(points,colors)
    return points


def sample_subcommunity(points,nns,color,local=False):
    similarities = compute_euclidian(np.array(points))
    if not local:
        sample = np.random.choice(range(len(points)),nns)
    else:
        random_point = random.choice(range(len(points)))
        plt.plot(points[random_point][0], points[random_point][1], 'x', label = 'xy', color='black', ms=8)
        sample_dis = np.array(similarities[random_point])
        sample = np.argsort(-sample_dis)[:nns]
    community = [points[i] for i in sample]
    plot(np.array(community),color,3,linked=False)
    c = centroid(np.array(community))
    plt.plot(c[0], c[1], 'o', label = 'xy', color='black', ms=15)
    return community


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, IFS 0.1')
    print(args)

    plt.figure(figsize=(20,5))
    theta = float(args["--translate"])
    iterations = int(args["--iter"])
    sample_n = int(args["--sample"])
    cmap=get_cmap(iterations)

    A = mk_attractor()

    plt.subplot(131)
    colors=['maroon','gold','red','olivedrab']
    all_points = mk_color_points_dict(colors)
    for i in range(A.shape[0]):
        points = run_attractor(A,i,theta,iterations,colors)
        all_points[i].extend(points[i])
    plot_attractor(A)
    
    plt.subplot(132)
    for i in range(A.shape[0]):
        sample_subcommunity(all_points[i],sample_n,colors[i],local=False)
    
    plt.subplot(133)
    for i in range(A.shape[0]):
        sample_subcommunity(all_points[i],sample_n,colors[i],local=True)
    plt.show()
