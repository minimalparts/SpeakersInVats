import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

def run_PCA(dm_mat):
    pca = PCA(n_components=2)
    pca.fit(dm_mat)
    m_2d = pca.transform(dm_mat)
    return np.array(m_2d)

def run_STNE(dm_mat):
    m_2d = TSNE(n_components=2).fit_transform(dm_mat)
    return np.array(m_2d)

def run_MDS(dm_mat):
    m_2d = MDS(n_components=10).fit_transform(dm_mat)
    return np.array(m_2d)

def readCols(infile):
    with open(infile) as f:
        cols=f.read().splitlines() 
        f.close()
    return cols

def readDM(infile):
    '''Read dm file'''
    word_to_i = {}
    i_to_word = {}

    with open(infile) as f:
        dmlines=f.readlines()
        f.close()

    '''Make numpy array'''
    vectors = []
    i = 0
    for l in dmlines:
        items=l.rstrip('\n').split('\t')
        row=items[0]
        vec=np.array([float(c) for c in items[1:]])
        vectors.append(vec)
        word_to_i[row] = i
        i_to_word[i] = row
        i+=1  
    dm_mat = np.vstack(vectors)
    print(dm_mat.shape)
    return dm_mat, word_to_i, i_to_word

def printDM(dm_mat,outfile,i_to_word):
    '''Print new dm file'''
    f = open(outfile,'w')
    for c in range(dm_mat.shape[0]):
        vec = '\t'.join([str(i) for i in dm_mat[c]])
        f.write(i_to_word[c]+"\t"+vec+"\n")
    f.close()

