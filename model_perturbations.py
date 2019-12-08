"""Model perturbations with rotations and scaling

Usage:
model_perturbations.py --control=<file> --dir=<d> --v=<param_value> --locus=<n> --num_words=<n>  --nns=<n> [--viz]
model_perturbations.py --version

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
from utils.utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, rmse, get_vocab_freqs, percentile, average
from utils.evals import RSA, compute_cosines, compute_nearest_neighbours
from utils.transformations import center, scale, rotate, find_svd_rotation

np.random.seed(0)

def unpack(f,vatdir):
    shutil.unpack_archive(f, extract_dir=vatdir)

def make_figure(m, words):
    n = int(m.shape[0] / 3)
    plt.plot(m[:n, 0], m[:n, 1], 'o-', label = 'control', color='red')
    for i in range(len(words)):
        plt.annotate(words[i], xy=(m[i][0], m[i][1]), xytext=(-5, 10), textcoords='offset points', color='red', size=10)
    plt.savefig("img/"+words[0]+"_control.png")
    plt.clf()
    plt.plot(m[n:n*2, 0], m[n:n*2, 1], 'o-', label = 'transformed', color='blue')
    for i in range(len(words)):
        plt.annotate(words[i], xy=(m[i+n][0], m[i+n][1]), xytext=(-5, 10), textcoords='offset points', color='blue', size=10)
    plt.savefig("img/"+words[0]+"_transformed.png")
    plt.clf()
    plt.plot(m[n*2:n*3, 0], m[n*2:n*3, 1], 'o-', label = 'perturbed', color='green')
    for i in range(len(words)):
        plt.annotate(words[i], xy=(m[i+2*n][0], m[i+2*n][1]), xytext=(-5, 10), textcoords='offset points', color='green', size=10)
    plt.savefig("img/"+words[0]+"_perturbed.png")
    plt.clf()

def read_params(d):
    value = 0.0
    locus = None
    f = open(join(d,"settings.txt"))
    for l in f:
        l = l.rstrip('\n')
        if l[:4] == '--v ':
            value = l.split()[1]
        if l[:8] == '--locus ':
            locus = l.split()[1]
    return value, locus


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
    #print("Processing control speaker...")
    m = process_matrix(m,40)
    neighbourhood = nns(m,vocab,word,num_nns)
    indices = [vocab.index(w) for w in neighbourhood]
    nn_vectors = [m[i] for i in indices] 
    return neighbourhood, nn_vectors

def get_speaker_data(vatdir,neighbourhood,v,l):
    nn_vectors = None
    speaker_files = [join(vatdir,f) for f in listdir(vatdir)]
    for sfile in speaker_files:
        unpack(sfile,vatdir)
        vat = sfile.replace(".zip","")
        value,locus = read_params(vat)

        if value == v and locus == l:
            print("Processing",sfile,"...")
            m,vocab = read_external_vectors(join(vat,"s0.dm"))
            #m = process_matrix(m,40)
            indices = [vocab.index(w) for w in neighbourhood]
            nn_vectors = [m[i] for i in indices] 
            shutil.rmtree(vat)
            break
        shutil.rmtree(vat)
    return nn_vectors

def select_words(m,vocab,locus,n):
    locus = int(locus)
    freqs = np.array(get_vocab_freqs(m,vocab))
    indices = percentile(freqs,locus)
    indices = [random.choice(indices) for c in range(n)] 
    return [vocab[i] for i in indices]

def find_scale(control,perturbed):
    all_scales = {}

    for i in range(1,50):
        j = 1+i/50
        d = rmse(scale(control,j), perturbed)
        all_scales[j] = d

    for i in range(1,50):
        k = 1-i/50
        d = rmse(scale(control,k), perturbed)
        all_scales[k] = d

    minkey = min(all_scales, key=all_scales.get)
    return minkey, all_scales[minkey]
    


if __name__=="__main__":
    args = docopt(__doc__, version='Semantic chaos, model_perturbations 0.1')
    print(args)

    vatdir = args["--dir"]
    value = args["--v"]
    locus = args["--locus"]
    viz = args["--viz"]
    control_file = args["--control"]
    num_nns = int(args["--nns"])
    num_words = int(args["--num_words"])

    ref,vocab = read_external_vectors(control_file)
    words = select_words(ref,vocab,locus,num_words)
    print("\nTEST WORDS:",words,"\n")

    avg_perturbed_control = []
    avg_perturbed_rotation = []
    avg_perturbed_transformed = []

    for word in words:
        control_neighbourhood, control_nn_vectors = get_reference_data(ref,vocab,word,num_nns)
        print("NEIGHBOURHOOD:",control_neighbourhood)
        control = center(np.array(control_nn_vectors))
        nn_vectors = get_speaker_data(vatdir,control_neighbourhood,value,locus)
        perturbed = center(np.array(nn_vectors))
        avg_perturbed_control.append(rmse(control,perturbed))

        '''Rotation'''
        print("ROTATING")
        transformed = find_svd_rotation(control,perturbed)
        avg_perturbed_rotation.append(rmse(transformed,perturbed))

        '''Scaling'''
        best_scale,distance = find_scale(transformed,perturbed)
        print("SCALING (FACTOR:",best_scale,distance,")")
        transformed = scale(transformed,best_scale)
        avg_perturbed_transformed.append(rmse(transformed,perturbed))

        if viz:
            concatenated = np.concatenate((control, transformed, perturbed))    
            concatenated_2d = compute_PCA(concatenated,2)
            make_figure(concatenated_2d, control_neighbourhood)
        

print("AVG RMSE, CONTROL - PERTURBED:", average(avg_perturbed_control))
print("AVG RMSE, CONTROL - ROTATED:", average(avg_perturbed_rotation))
print("AVG RMSE, CONTROL - ROTATED+SCALED:", average(avg_perturbed_transformed))
