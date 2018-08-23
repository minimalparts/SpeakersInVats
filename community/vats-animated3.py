import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from multiprocessing.dummy import Pool as ThreadPool 
from itertools import repeat
from utils import mk_space, run_STNE, run_PCA, read_matrix, euclidian_distance, nearest_neighbours, sim_matrix2

np.set_printoptions(precision=3)
pool = ThreadPool(2)
plt.ion()
plt.show()

pointsize = 24
np.random.seed(0)

def get_cmap():
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.tab10(vals))
    return cmap

def perturb(point1, point2):
    '''Given two points, get new point between them
    typically closer to point1 than point2.'''
    #i = np.random.randint(20,30)
    i = 10
    new_point =  [point1[0] + (point2[0]-point1[0])/i, point1[1] + (point2[1]-point1[1])/i]
    #new_point =  [point1[0] + (point2[0]-point1[0])*i, point1[1] + (point2[1]-point1[1])*i]
    #print(point1,point2,new_point,i)
    return new_point


def perturb_space(all_spaces,all_distances,n,cmap,vocab):
    '''Make new space as perturbation of existing space.
       For every vector in space, select some nearest neighbour
       in range n and move the vector towards that neighbour.'''
    new_space = []
    for i in range(all_spaces[0].shape[0]):
        s = np.random.choice(range(len(all_spaces)))
        #print("Perturbing in space",s)
        space = all_spaces[s]
        distances = all_distances[s]
        #nns = nearest_neighbours(distances,i,n)
        #r = np.random.choice(nns)
        r = np.random.randint(0,space.shape[0])
        #print("Perturbing",vocab[i],"in direction",vocab[r])
        v = perturb(space[i],space[r])
        #print(space[i],v)
        plt.scatter(v[0],v[1],color=cmap(i), s=pointsize)
        if i%20 == 0:
            plt.pause(0.01)
        new_space.append(v)
    return np.array(new_space)

def compute_rho(distances1,distances2):
    rho,p = stats.spearmanr(distances1.flatten(),distances2.flatten())
    return rho

def avg_rho(all_distances,distances1):
    '''Calculate correlation of some space to all existing
       spaces. This simulates whether the speaker represented
       by the new space could talk to all other speakers.'''
    rhos_over_spaces = pool.starmap(compute_rho, zip(repeat(distances1),all_distances))
    #print(rhos_over_spaces)
    return rhos_over_spaces


if __name__=="__main__":
    num_speakers = 200
    cmap = get_cmap()
    space, word_to_i, i_to_word = read_matrix(sys.argv[1])
    vocab = [ i_to_word[c] for c in range(len(word_to_i))]
    already_annotated = []
    for label, x, y in zip(vocab, space[:, 0], space[:, 1]):
        if [x,y] not in already_annotated:
            plt.annotate(label,xy=(x, y), xytext=(-15, 15), textcoords='offset points', size=16)
            already_annotated.append([x,y])
    distances = sim_matrix2(space)
    rhos = []

    all_spaces = [space]
    all_distances = [distances]
    for p in range(num_speakers-1):
        print("Creating speaker",p+1)
        new_space = perturb_space(all_spaces,all_distances,100,cmap,vocab)
        new_distances = sim_matrix2(new_space)
        rhos = avg_rho(all_distances,new_distances)
        #print(rhos)
        if any([r < 0.6 for r in rhos]):
            print("Space is bad")
            continue
        else:
            all_spaces.append(new_space)
            all_distances.append(new_distances)
    plt.show(block=True)
    print(len(all_spaces),"remaining...")

