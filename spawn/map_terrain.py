"""Map terrain around each perturbed neighbourhood

Usage:
map_terrain.py --control=<file> --dir=<d>  --locus=<n> --v=<n>
map_terrain.py --version

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
from utils import read_external_vectors, ppmi, normalise_l2, compute_PCA, rmse, get_vocab_freqs, percentile, average, get_vocab_freqs, compute_cosines
from evals import compute_men_spearman, RSA, compute_cosines, compute_nearest_neighbours
from transformations import center, scale, rotate, find_svd_rotation

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

def nns(cosines,vocab,word,num_nns):
    word_indices = {}
    for i in range(len(vocab)):
        word_indices[i] = vocab[i]
    nns = compute_nearest_neighbours(cosines,word_indices,vocab.index(word),num_nns)
    return [nn[0] for nn in nns], [nn[1] for nn in nns]

def get_reference_data(m,vocab,cosines,word,num_nns):
    #print("Processing control speaker...")
    neighbourhood, cosines = nns(cosines,vocab,word,num_nns)
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
            #print("Processing",sfile,"...")
            m,vocab = read_external_vectors(join(vat,"s0.dm"))
            cosines = compute_cosines(m)
            men = round(compute_men_spearman(m,vocab)[0],2)
            indices = [vocab.index(w) for w in neighbourhood]
            nn_vectors = [m[i] for i in indices] 
            shutil.rmtree(vat)
            break
        shutil.rmtree(vat)
    return nn_vectors, men, cosines


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


def isolation(m,vocab,neighs):
    iso = 0.0
    for nn in neighs:
        neighbourhood, cosines = nns(m,vocab,nn,len(neighs)+1)
        for w in neighbourhood:
            if w not in neighs:
                iso+=cosines[neighbourhood.index(w)]
                break
    return 1 - iso / len(neighs)
        


def write_entry(nns,men,rsa,rmse1,rmse2,rmse3,scalef,wordfreq,density,isolation):
    neigh = ' '.join(n for n in nns)
    print(nns[0],"\tMEN:",men,"\tRSA",rsa,"\tRMSE ORIG:",rmse1,"\tRMSE R:",rmse2,"\tRMSE FINAL:",rmse3,"\tSCALING:",scalef,"\tFREQ:",int(wordfreq),"\tDENSITY:",density,"\tISOLATION:",isolation,"\tNEIGH:",neigh)


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    print(args)

    vatdir = args["--dir"]
    locus = args["--locus"]
    value = args["--v"]
    control_file = args["--control"]

    num_nns = [2,5,10,20,50]

    ref,vocab = read_external_vectors(control_file)
    freqs = get_vocab_freqs(ref,vocab)
    ref = process_matrix(ref,40)
    ref_cosines = compute_cosines(ref)

    for i in range(len(vocab)):
        for num_nn in num_nns:
            control_neighbourhood, control_nn_vectors = get_reference_data(ref,vocab,ref_cosines,vocab[i],num_nn)
            control = center(np.array(control_nn_vectors))
            l = len(control_nn_vectors)
            density = np.sum(compute_cosines(control_nn_vectors)) / (l*l)
            iso = isolation(ref,vocab,control_neighbourhood)

            nn_vectors, men, pert_cosines = get_speaker_data(vatdir,control_neighbourhood,value,locus)
            rsa = RSA(ref_cosines, pert_cosines)[0]
            perturbed = center(np.array(nn_vectors))
            rmse1 = rmse(control,perturbed)

            '''Rotation'''
            transformed = find_svd_rotation(control,perturbed)
            rmse2 = rmse(transformed,perturbed)

            '''Scaling'''
            best_scale,distance = find_scale(transformed,perturbed)
            transformed = scale(transformed,best_scale)
            rmse3 = rmse(transformed,perturbed)

            write_entry(control_neighbourhood,men,rsa,rmse1,rmse2,rmse3,best_scale,freqs[i],density,iso)
