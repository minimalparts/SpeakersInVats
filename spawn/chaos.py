"""Visualise 2D IFS

Usage:
chaos.py --control=<file> --nns=<n> --coinbias=<n> [--translate=<n>] [--print_zeros]
chaos.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import random
import numpy as np
from docopt import docopt
from math import cos,sin
from transformations import rotate, center
from evals import compute_men_spearman, RSA, compute_cosines, compute_nearest_neighbours
from utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, rmse


def mk_attractor(control_file):
    m, vocab = read_external_vectors(control_file)
    return m, vocab

def process_matrix(m,dim):
    m = ppmi(m)
    m = normalise_l2(m)
    m = compute_PCA(m,dim)
    return m

def linear(p,lambd):
    '''Scaling with lambda< 1 decreases all frequencies.'''
    #print("Scaling...")
    p = lambd * p
    p = np.array([int(i) for i in p])
    return p

def translation(p,att,theta):
    #print("Translating...")
    for i in range(len(p)):
        t = theta * (p[i] - att[i])
        p[i] = p[i] - t
    p = np.array([int(i) for i in p])
    return p

def rotation(v,rho):
    #print("Rotating...")
    v = np.array([v])
    for i in range(len(v[0])-1):
        vt = rotate(v,i,i+1,rho)
        if vt[0][i] < 0 or vt[0][i+1] < 0:
            vt = rotate(v,i,i+1,-rho)
    v = np.array([int(i) for i in v[0]])
    return v

def find_zero_cells(m):
    return np.argwhere(m == 0)

def print_perturbed_zeros(A,B):
    orig_zero_cells = set([tuple(p) for p in find_zero_cells(A)])
    spawned_zero_cells = set([tuple(p) for p in find_zero_cells(B)])
    perturbed_zeros = list(orig_zero_cells - spawned_zero_cells)
   
    print("NUM ORIG ZEROS:",len(orig_zero_cells), "NUM SPAWNED ZEROS:",len(spawned_zero_cells), "NUM PERTURBED ZEROS:",len(perturbed_zeros))
    some_perturbed_zeros = random.sample(perturbed_zeros,20)
    for z in some_perturbed_zeros:
         print(vocab[z[0]],vocab[z[1]],B[z[0]][z[1]])

def mk_community(A,vocab):
    C = {}
    for i in range(len(vocab)):
        C[i] = [A[i]]
    return C

def sample_community(C):
    sample = []
    for i in range(len(C)):
        ind = random.randint(0,len(C[i])-1)
        sample.append(C[i][ind])
    return np.array(sample)

def coinflip(prob):
    if random.random() < prob:
        return True
    else:
        return False


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, IFS 0.1')
    print(args)

    control_file = args["--control"]
    nns = int(args["--nns"])
    bias = float(args["--coinbias"])
    if args["--translate"]:
        theta = float(args["--translate"])

    A, vocab = mk_attractor(control_file)
    A_processed = process_matrix(A,40)
    A_cosines = compute_cosines(A_processed)
    B = A.copy()
    C = mk_community(A, vocab)
    
    for i in range(len(A)):
        p = A.copy()[i]    #copy important -- otherwise changing A itself while changing p!
        while True:
            #print(i,vocab[i],compute_nearest_neighbours(A_cosines,vocab,i,10))
            idx = (-A_cosines[i]).argsort()[:nns]
            att = np.random.choice(idx)
            print(vocab[i],"-->",vocab[att],A_cosines[i][att])
            p = translation(p,A[att],theta)
            C[i].append(p)
            if coinflip(bias):
                B[i] = p
                break

    A_processed = process_matrix(A,40)
    A_cosines = compute_cosines(A_processed)
    B_processed = process_matrix(B,40)
    B_cosines = compute_cosines(B_processed)

    print("SPEARMAN CONTROL:",compute_men_spearman(A_processed,vocab))
    print("SPEARMAN SPAWNED:",compute_men_spearman(B_processed,vocab))
    print("RSA",RSA(A_cosines,B_cosines))
    print("RMSE RAW SPACES:",rmse(A,B))
    print("RMSE LATENT SPACES:",rmse(A_processed,B_processed))
    print("CONTROL NNS:",compute_nearest_neighbours(A_cosines,vocab,vocab.index('dog'),5))
    print("SPAWNED NNS:",compute_nearest_neighbours(B_cosines,vocab,vocab.index('dog'),5))
    print("\n")

    if (args["--print_zeros"]):
        print_perturbed_zeros(A,B)

    for i in range(10):
        print("SAMPLED SPEAKER",i)
        S = sample_community(C)
        S_processed = process_matrix(S,40)
        S_cosines = compute_cosines(S_processed)
        print("SPEARMAN SPAWNED:",compute_men_spearman(S_processed,vocab))
        print("RSA",RSA(A_cosines,S_cosines))


