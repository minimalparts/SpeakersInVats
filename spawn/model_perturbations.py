"""Visualise nearest neighbours

Usage:
visualise_nns.py --control=<file> --dir=<d> --v=<param_value> --locus=<n>
visualise_nns.py --version

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
from utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, matrix_distance, get_vocab_freqs, percentile
from evals import RSA, compute_cosines, compute_nearest_neighbours
from transformations import center, stretch

np.random.seed(0)

def unpack(f,vatdir):
    shutil.unpack_archive(f, extract_dir=vatdir)


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

def nns(m,vocab,word):
    cosines = compute_cosines(m)
    word_indices = {}
    for i in range(len(vocab)):
        word_indices[i] = vocab[i]
    nns = compute_nearest_neighbours(cosines,word_indices,vocab.index(word))
    return [nn[0] for nn in nns]

def get_reference_data(m,vocab,word):
    print("Processing control speaker...")
    print(m.shape)
    m = process_matrix(m,40)
    neighbourhood = nns(m,vocab,word)
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
            indices = [vocab.index(w) for w in neighbourhood]
            nn_vectors = [m[i] for i in indices] 
            shutil.rmtree(vat)
            break
        shutil.rmtree(vat)
    return nn_vectors

def select_words(m,vocab,locus):
    locus = int(locus)
    freqs = np.array(get_vocab_freqs(m,vocab))
    indices = percentile(freqs,locus)
    indices = [random.choice(indices) for c in range(5)] 
    return [vocab[i] for i in indices]

def find_stretch(control,perturbed):
    all_stretches = {}

    for i in range(1,50):
        j = 1+i/50
        d = matrix_distance(stretch(control,j), perturbed)
        all_stretches[j] = d

    for i in range(1,50):
        k = 1-i/50
        d = matrix_distance(stretch(control,k), perturbed)
        all_stretches[k] = d

    minkey = min(all_stretches, key=all_stretches.get)
    return minkey, all_stretches[minkey]
    

if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    print(args)

    vatdir = args["--dir"]
    value = args["--v"]
    locus = args["--locus"]
    control_file = args["--control"]

    ref,vocab = read_external_vectors(control_file)
    words = select_words(ref,vocab,locus)
    print(words)

    for word in words:
        control_neighbourhood, control_nn_vectors = get_reference_data(ref,vocab,word)
        print(control_neighbourhood)
        control = center(np.array(control_nn_vectors))
        nn_vectors = get_speaker_data(vatdir,control_neighbourhood,value,locus)
        perturbed = center(np.array(nn_vectors))
        print(matrix_distance(control, perturbed))
        best_stretch,distance = find_stretch(control,perturbed)
        print("BEST STRETCH:",best_stretch,distance)


