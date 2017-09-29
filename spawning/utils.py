import random
import numpy as np


def get_rand_freq(word_freqs):
    '''Return a random frequency out of vocabulary'''
    return word_freqs[random.choice(word_freqs.keys())]

def get_rand_int_vec(dim,max_sum):
    print dim,max_sum
    r = [random.random() for i in range(dim)]
    r = [round(max_sum*i/sum(r)) for i in r]
    return r

def compute_freq_diff(original,new):
    fo = original.sum()
    fn = new.sum()
    print fo, fn
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

def readDM(infile):
    '''Read dm file'''
    word_to_i = {}
    i_to_word = {}
    word_freqs = {}
    word_vars = {}

    with open(infile) as f:
        dmlines=f.readlines()
        f.close()

    '''Make numpy array'''
    i = 0
    first_line = dmlines[0].rstrip('\n').split('\t')
    word_to_i[first_line[0]] = i
    i_to_word[i] = first_line[0]
    i+=1
    dm_mat = np.array([float(c) for c in first_line[1:]])
    dm_mat = dm_mat.reshape(1,len(dm_mat))
    word_freqs[first_line[0]] = int(dm_mat[0].sum())
    word_vars[first_line[0]] = int(round(np.var(dm_mat[0])))
    for l in dmlines[1:]:
        items=l.rstrip('\n').split('\t')
        row=items[0]
        vec=np.array([float(c) for c in items[1:]])
        #print row,int(vec.max()), int(vec.min()), int(vec.sum()), int(round(np.var(vec)))
        word_freqs[row] = int(vec.sum())
        word_vars[row] = int(round(np.var(vec)))
        dm_mat = np.vstack([dm_mat,vec])
        word_to_i[items[0]] = i
        i_to_word[i] = items[0]
        i+=1  
    max_freq = word_freqs[max(word_freqs, key=word_freqs.get)]
    print dm_mat.shape
    return dm_mat, word_to_i, i_to_word, word_freqs, max_freq

def printDM(dm_mat,outfile,i_to_word):
    '''Print new dm file'''
    f = open(outfile,'w')
    for c in range(dm_mat.shape[0]):
        vec = '\t'.join([str(i) for i in dm_mat[c]])
        f.write(i_to_word[c]+"\t"+vec+"\n")
    f.close()

