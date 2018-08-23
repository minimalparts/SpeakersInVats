import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#Define a few random vectors.
def mk_space(cmap, pointsize):
    '''Make new random 2D space.'''
    space = []
    for i in range(50):
        v = np.random.rand(2)
        space.append(v)
        plt.scatter(v[0],v[1],color=cmap(i),s=pointsize)
        if i%20 == 0:
            plt.pause(0.01)
    return np.array(space)

#def readDM(dm_file, cmap, pointsize):
def readDM(dm_file):
    dm_dict = {}
    version = ""
    with open(dm_file) as f:
        dmlines=f.readlines()
    f.close()

    #Make dictionary with key=row, value=vector
    for l in dmlines:
        items=l.rstrip().split()
        row=items[0]
        vec=[float(i) for i in items[1:]]
        vec=np.array(vec)
        dm_dict[row]=vec
    return dm_dict

def read_matrix(infile):
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
        items=l.rstrip('\n').split()
        row=items[0]
        vec= [float(c) for c in items[1:]]
        keep = True
        for v in vectors:
            if all( vec[i] == v[i] for i in range(len(v)) ):
                keep = False
        if keep or len(vectors) == 0:
            vectors.append(vec)
            word_to_i[row] = i
            i_to_word[i] = row
            i+=1  
    dm_mat = np.vstack(np.array(vectors))
    print(dm_mat[:10], word_to_i)
    return dm_mat, word_to_i, i_to_word

def run_PCA(dm_dict, words):
    m = []
    labels = []
    for w in words:
        labels.append(w)
        m.append(dm_dict[w])
    pca = PCA(n_components=2)
    pca.fit(m)
    m_2d = pca.transform(m)
    return m_2d

def run_PCA(m):
    pca = PCA(n_components=2)
    pca.fit(m)
    m_2d = pca.transform(m)
    return m_2d

def run_STNE(dm_dict, words):
    m = []
    labels = []
    for w in words:
        labels.append(w)
        m.append(dm_dict[w])
    m_2d = TSNE(n_components=2).fit_transform(m)
    return np.array(m_2d)


def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))

def euclidian_distance(v1,v2):
    return np.linalg.norm(v1-v2)

def nearest_neighbours(distances,v,n):
    return distances[v].argsort()[-3:][::-1]

def sim_matrix(space):
    '''Distances between every pair of words in space.'''
    distances = []
    for i in range(space.shape[0]):
        distances.append([])
        for j in range(space.shape[0]):
            d = euclidian_distance(space[i],space[j])
            #print(space[i],space[j],cos)
            distances[i].append(d)
    distances = np.array(distances)
    #print(distances)
    return distances

def sim_matrix2(space):
    space2 = space.reshape(space.shape[0], 1, space.shape[1])
    distances = np.sqrt(np.einsum('ijk, ijk->ij', space-space2, space-space2))
    #print(distances)
    return distances

def top_indices(distances, n):
    top = np.zeros(n)
    top_i = [[],[],[]]
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            min_i = np.argmin(top)
            if i != j and distances[i][j] > top[min_i]:
                #print(i,j,distances[i][j],min_i,top[min_i])
                top[min_i] = distances[i][j]
                top_i[min_i] = [i,j]
    #print(top,top_i)                
