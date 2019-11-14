"""Observe effect of initial conditions

Usage:
chaos.py --control=<file> --word=<s> --rotate=<param_value> --scale=<param_value> --nns=<n>
chaos.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import sys
import shutil
import numpy as np
import random
from docopt import docopt
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, rmse, get_vocab_freqs, percentile, average
from evals import RSA, compute_cosines, compute_nearest_neighbours
from transformations import center, scale, rotate, find_svd_rotation

np.random.seed(0)

def make_figure(m, num_nns):
    num_nns+=1
    n = int(m.shape[0] / num_nns)
    print(n)
    plt.plot(m[:num_nns, 0], m[:num_nns, 1], 'o', label = 'control', color='red', ms=10)
    #for i in range(1,int(n/2)):
    for i in range(1,n):
        plt.plot(m[num_nns*i:num_nns*i+num_nns, 0], m[num_nns*i:num_nns*i+num_nns, 1], 'o', color='blue')
    #for i in range(int(n/2)+1,n):
    #    plt.plot(m[num_nns*i:num_nns*i+num_nns, 0], m[num_nns*i:num_nns*i+num_nns, 1], 'o', color='green')
    plt.show()

def process_matrix(m,dim):
    m = ppmi(m)
    m = normalise_l2(m)
    m = compute_PCA(m,dim)
    return m

def nns(m,vocab,word,num_nns):
    cosines = compute_cosines(m)
    word_indices = {}
    for i in range(len(vocab)):
        word_indices[i] = vocab[i]
    nns = compute_nearest_neighbours(cosines,word_indices,vocab.index(word),num_nns)
    return [nn[0] for nn in nns]

def get_reference_data(m,vocab,word,num_nns):
    print("Processing control speaker...")
    m = process_matrix(m,40)
    neighbourhood = nns(m,vocab,word,num_nns)
    indices = [vocab.index(w) for w in neighbourhood]
    nn_vectors = [m[i] for i in indices] 
    return neighbourhood, nn_vectors


    


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, chaos 0.1')
    print(args)

    control_file = args["--control"]
    word = args["--word"]
    vr = int(args["--rotate"])
    vs = float(args["--scale"])
    num_nns = int(args["--nns"])

    ref,vocab = read_external_vectors(control_file)
    control_neighbourhood, control_nn_vectors = get_reference_data(ref,vocab,word,num_nns)
    control = center(np.array(control_nn_vectors))
    control2 = control.copy()
    noise = np.random.normal(0,0.001,control2.shape)
    control2+=noise
    print(control_neighbourhood)
    print(control[0][:10])
    print(control2[0][:10])

    transformed = control.copy()
    transformed2 = control2.copy()
    concatenated = control.copy()
    concatenated2 = control.copy()

    for k in range(5):
        for i in range(40-1):
  
            '''Rotation'''
            transformed = rotate(transformed, i, i+1, vr)
            transformed2 = rotate(transformed2, i, i+1, vr)
            print("ROT:",rmse(transformed,control))
            print("ROT2:",rmse(transformed2,control))
            print("1/2:",rmse(transformed,transformed2))

            '''Scaling'''
            transformed = scale(transformed,vs)
            transformed2 = scale(transformed2,vs)
            print("SCALE:",rmse(transformed,control))
            print("SCALE2:",rmse(transformed2,control))
            print("1/2:",rmse(transformed,transformed2))
        
            concatenated = np.concatenate((concatenated, transformed))    
            concatenated2 = np.concatenate((concatenated2, transformed2))    
    
    concatenated = np.concatenate((concatenated, concatenated2))       
    concatenated_2d = compute_PCA(concatenated,2)
    make_figure(concatenated_2d, num_nns)
