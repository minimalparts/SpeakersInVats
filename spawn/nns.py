"""Return nearest neighbours

Usage:
nns.py --control=<file> --dir=<d> --locus=<n> --v=<n> --word=<s>
nns.py --version

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
from utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, compute_cosines
from evals import compute_men_spearman, RSA, compute_cosines, compute_nearest_neighbours

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


def get_reference_data(m,vocab,cosines,word,num_nns):
    #print("Processing control speaker...")
    neighbourhood, cosines = nns(cosines,vocab,word,num_nns)
    indices = [vocab.index(w) for w in neighbourhood]
    nn_vectors = [m[i] for i in indices] 
    return neighbourhood, nn_vectors

def get_speaker_data(vatdir,v,l):
    m = None
    speaker_files = [join(vatdir,f) for f in listdir(vatdir)]
    for sfile in speaker_files:
        unpack(sfile,vatdir)
        vat = sfile.replace(".zip","")
        value,locus = read_params(vat)

        if value == v and locus == l:
            print("Processing",sfile,"...")
            m,vocab = read_external_vectors(join(vat,"s0.dm"))
            shutil.rmtree(vat)
            break
        shutil.rmtree(vat)
    return m




if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    print(args)

    vatdir = args["--dir"]
    control_file = args["--control"]
    locus = args["--locus"]
    v = args["--v"]
    word = args["--word"]

    ref,vocab = read_external_vectors(control_file)
    ref = process_matrix(ref,40)
    m = get_speaker_data(vatdir,v,locus)
    ref_cosines = compute_cosines(ref)
    m_cosines = compute_cosines(m)

    print("SPEARMAN SPAWNED:",compute_men_spearman(m,vocab))
    print("RSA:",RSA(ref_cosines,m_cosines))
    print("CONTROL NNS:",compute_nearest_neighbours(ref_cosines,vocab,vocab.index(word),5))
    print("SPAWNED NNS:",compute_nearest_neighbours(m_cosines,vocab,vocab.index(word),5))
