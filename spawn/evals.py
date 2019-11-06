import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

def compute_men_spearman(m,vocab):
    system = []
    gold = []

    f = open("data/MEN_dataset_natural_form_full",'r')
    for l in f:
        fields = l.rstrip('\n').split()
        w1 = fields[0]
        w2 = fields[1]
        score = float(fields[2])
        if w1 in vocab and w2 in vocab:
            cos = 1 - distance.cosine(m[vocab.index(w1)],m[vocab.index(w2)])
            system.append(cos)
            gold.append(score)
    f.close()

    #print(len(gold),"pairs considered...")
    return spearmanr(system,gold)

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

def RSA(m1_cos,m2_cos):
    top_m1 = upper_tri_indexing(m1_cos).flatten()
    top_m2 = upper_tri_indexing(m2_cos).flatten()
    return spearmanr(top_m1,top_m2)

def compute_cosines(m):
    return 1-pairwise_distances(m, metric="cosine")

def compute_nearest_neighbours(cosines,word_indices,i):
    word_cos = np.array(cosines[i])
    ranking = np.argsort(-word_cos)
    neighbours = [(word_indices[n], round(word_cos[n],5)) for n in ranking][:5]
    return neighbours
