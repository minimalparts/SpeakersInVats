import sys
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np


system = []
gold = []


def read_predicate_matrix(loc):
    vocab = []
    vectors = []
    print(loc)
    with open(loc) as f:
        dmlines=f.read().splitlines()
    for l in dmlines:
        items=l.split()
        target = items[0]
        vocab.append(target)
        vec=[float(i) for i in items[1:]]       #list of lists  
        vectors.append(vec)
    m = np.array(vectors)
    return vocab, m

def normalise_l2(m):
    return preprocessing.normalize(m, norm='l2')

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



vocab, m = read_predicate_matrix(sys.argv[1])
m = ppmi(m)
m = normalise_l2(m)
m = compute_PCA(m,10)
f = open("MEN_dataset_natural_form_full",'r')

for l in f:
    fields = l.rstrip('\n').split()
    w1 = fields[0]
    w2 = fields[1]
    score = float(fields[2])
    if w1 in vocab and w2 in vocab:
        cos = 1 - distance.cosine(m[vocab.index(w1)],m[vocab.index(w2)])
        system.append(cos)
        gold.append(score)
        print(w1,w2,cos,score)
f.close()

print("SPEARMAN:",spearmanr(system,gold))
print("("+str(len(system))+" pairs out of the original 3000 could be processed, due to vocabulary size.)")

