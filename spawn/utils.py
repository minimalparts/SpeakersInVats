import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from scipy import linalg as LA
from os.path import join
from math import log
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_rand_freq(word_freqs,min_freq):
    '''Return a random frequency out of vocabulary -- following zipf'''
    freq = 0.0
    freqs = [ v for k,v in word_freqs.items() if v >= min_freq ]
    print(max(word_freqs.values()),min_freq,freqs[:10])
    while freq == 0.0:
        freq = random.choice(freqs)
    return freq

def get_rand_ints_with_sum(length,total):
    v = np.zeros(length)
    remainder = total
    position = 0
    while remainder > 0:
        r = random.randint(0, remainder)
        v[position]+=r
        remainder -= r
        position+=1
        if position == len(v):
            position = 0
    return v

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
            if row_sums[i] == 0 or row_sums[j] == 0:
                print("WARNING: encountered zero vector.")
                ppmi_matrix[i][j] = 0
            else:
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


def get_vocab_ranked_logs(m,vocab):
    freqs = {}
    for w in vocab:
        freq = np.sum(m[vocab.index(w)])
        freqs[w] = freq
    c = 1
    log_ranks = []
    log_freqs = []
    for w in sorted(freqs, key=freqs.get, reverse=True):
        log_ranks.append(log(c))
        log_freqs.append(log(freqs[w]))
        c+=1
    return log_ranks, log_freqs


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

def print_list_pair(l1,l2,outfile):
    f = open(outfile,'w')
    for i in range(len(l1)):
        f.write(str(l1[i])+' '+str(l2[i])+'\n')
    f.close()

def read_params(infile):
    d = {}
    f = open(infile)
    for l in f:
        l = l.rstrip('\n')
        fields = l.split()
        d[fields[0]] = fields[1]
    return d

def read_ranked_freqs(infile):
    l1 = []
    l2 = []
    f = open(infile)
    for l in f:
        l = l.rstrip('\n')
        fields = l.split()
        l1.append(float(fields[0]))
        l2.append(float(fields[1]))
    return l1,l2

