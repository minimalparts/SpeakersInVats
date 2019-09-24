import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from scipy import linalg as LA
from os.path import join
import numpy as np


def get_rand_freq(word_freqs):
    '''Return a random frequency out of vocabulary'''
    freq = 0.0
    while freq == 0.0:
        freq = word_freqs[random.choice(word_freqs.keys())]
    return freq

def get_rand_int_vec(dim,max_sum):
    print(dim,max_sum)
    r = [random.random() for i in range(dim)]
    r = [round(max_sum*i/sum(r)) for i in r]
    return r

def compute_freq_diff(fo,new):
    fn = new.sum()
    print("FO",fo, "FN",fn)
    diff = float(fn) / float(fo)
    return diff

def get_zero_positions(vec):
    zeros = []
    for i in range(len(vec)):
        if vec[i] == 0:
            zeros.append(i)
    return zeros

def fix_vec_freq(vec,diff,up):
    '''Get vec to correct frequency'''
    for i in range(diff):
        position = random.randint(0,len(vec)-1)
        if up:
             vec[position]+=1
        else:
             while vec[position]==0:			#Super-hack
                 position+=1
             vec[position]-=1
    return vec

def normalise(m):
    norm_matrix = np.zeros(m.shape)
    for i in range(m.shape[0]):
        norm_matrix[i] = m[i] / np.sum(m[i])
    return norm_matrix

def normalise_l1(m):
    return preprocessing.normalize(m, norm='l1')

def normalise_l2(m):
    return preprocessing.normalize(m, norm='l2')

def compute_cosines(m):
    return 1-pairwise_distances(m, metric="cosine")

def ppmi(m):
    ppmi_matrix = np.zeros(m.shape)
    N = np.sum(m)
    row_sums = np.sum(m, axis=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ppmi_matrix[i][j] = max(0, m[i][j] * N / (row_sums[i] * row_sums[j]))
    return ppmi_matrix

def compute_PCA(m,dim):
    m -= np.mean(m, axis = 0)
    pca = PCA(n_components=dim)
    pca.fit(m)
    return pca.transform(m)

def compute_truncated_SVD(m,dim):
    m -= np.mean(m, axis = 0)
    tsvd = TruncatedSVD(dim)
    tsvd.fit(m) 
    return tsvd.transform(m)



def read_external_vectors(vector_file):
    vocab = []
    vectors = []
    with open(vector_file) as f:
        dmlines=f.read().splitlines()
    for l in dmlines:
        items=l.split()
        target = items[0]
        vocab.append(target)
        vec=[float(i) for i in items[1:]]       #list of lists  
        vectors.append(vec)
    m = np.array(vectors)
    return m, vocab

def read_reference_vectors(vector_file):
    word_freqs = {}
    word_vars = {}
    vocab = []
    vectors = []
    with open(vector_file) as f:
        dmlines=f.read().splitlines()
    for l in dmlines:
        items=l.split()
        target = items[0]
        vocab.append(target)
        vec = np.array([float(i) for i in items[1:]])
        word_freqs[target] = int(vec.sum())
        word_vars[target] = int(round(np.var(vec)))
        vectors.append(vec)
        if np.sum(vec) == 0:
           print("WARNING",target,"HAS 0 SUM")
    m = np.array(vectors)
    return vocab, m, word_freqs, word_vars


def print_matrix(dm_mat,vocab,outfile):
    '''Print new dm file'''
    f = open(outfile,'w')
    for c in range(dm_mat.shape[0]):
        vec = ' '.join([str(i) for i in dm_mat[c]])
        f.write(vocab[c]+" "+vec+"\n")
    f.close()

def print_vocab(vocab,outfile):
    '''Print new vocab file'''
    f = open(outfile,'w')
    for c in range(len(vocab)):
        line = str(c)+' '+vocab[c]+'\n'
        f.write(line)
    f.close()

def print_dict(d,outfile):
    f = open(outfile,'w')
    for k,v in d.items():
        line = str(k)+' '+str(v)+'\n'
        f.write(line)
    f.close()

def print_list(l,outfile):
    f = open(outfile,'w')
    for v in l:
        f.write(str(v)+'\n')
    f.close()


