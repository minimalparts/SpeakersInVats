"""Create speakers by adding noise to a reference matrix

Usage:
spawn.py spawn <file_out>
spawn.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""


import sys
import math
import operator
import collections
import random
import numpy as np
import speaker_spearman, utils
from docopt import docopt


def beam_search(dm_mat,i_to_word,row_id,n):
    '''List of lists of highest probs (beam n)'''
    rows_highest = []
    '''Make row into probability distribution.'''
    row = dm_mat[row_id]
    p = row / row.sum()
    '''Get n highest probabilities.'''
    #words_uttered = np.random.choice(utils.init_int_list(len(row)),p=p, size=20)
    highest_probs = utils.largest_indices(p,n)[0]
    rows_highest.append(highest_probs)
    print i_to_word[row_id],[i_to_word[j] for j in highest_probs]
    for i in highest_probs:
        i_p = dm_mat[i] / dm_mat[i].sum()
        i_highest_probs = utils.largest_indices(i_p,n)[0]
        rows_highest.append(i_highest_probs)
        print i_to_word[i],[i_to_word[j] for j in i_highest_probs]
    c = collections.Counter(b for a in rows_highest for b in a)
    best_words = [a for a,b in c.items() if b >= 2]
    return best_words

def sample_matrix_random(dm_mat_file):
    dm_mat, word_to_i, i_to_word, word_freqs, max_freq = utils.readDM(dm_mat_file)
    print "Total number of words heard by speaker:", utils.total_freq(word_freqs)
    new_matrix = dm_mat.copy()
    new_total_words = 0
    for i in range(len(dm_mat[0])):
        '''Would the speaker talk to N about this?'''
        prob_topic = np.random.choice([0,1], p=[0.9, 0.1])
        if prob_topic == 1:
            #print "Speaker is chatting about",i_to_word[i]
            for j in range(i):
                new_matrix[i,j] = random.randint(0,dm_mat[i,j])
                new_matrix[j,i] = new_matrix[i,j]
                new_total_words+=new_matrix[i,j]
            #coocs = beam_search(dm_mat,i_to_word,i,20)
            #for i in coocs:
            #    print i_to_word[i]
        else:
            new_matrix[i] = np.zeros(len(new_matrix[i]))
    return new_matrix, i_to_word, new_total_words

def sample_matrix_random_bck(dm_mat_file):
    dm_mat, word_to_i, i_to_word, word_freqs, max_freq = utils.readDM(dm_mat_file)
    num_words_heard =  utils.total_freq(word_freqs)
    print "Total number of words heard by speaker:", num_words_heard
    new_matrix = np.zeros((len(word_to_i),len(word_to_i)))
    num_cells = dm_mat.flatten().size
    c = 0
    while c < num_words_heard / 10:
        c = np.random.randint(num_cells)
        if dm_mat[c/1000][c%1000] != 0:
            #print "hearing",i_to_word[c/1000],i_to_word[c%1000]
            new_matrix[c/1000][c%1000] !=1
            c+=1

def make_speaker():
    spawned_speaker, i_to_word, words_heard = sample_matrix_random("../spaces/speakers/speaker1.dm")
    num_words_heard = words_heard
    for i in range(2,11):
        speaker_file = "../spaces/speakers/speaker"+str(i)+".dm"
        new_matrix, i_to_word, words_heard = sample_matrix_random(speaker_file)
        spawned_speaker+=new_matrix
        num_words_heard+=words_heard
    print "NEW SPEAKER",num_words_heard
    return spawned_speaker, i_to_word



if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, spawn 0.1')
    print args


    if args["spawn"]:
        test_speaker, i_to_word = make_speaker()
        utils.printDM(test_speaker,args["<file_out>"], i_to_word)
        for i in range(1,11):
            spearman=speaker_spearman.run_spearman("../spaces/speakers/speaker"+str(i)+".dm",args["<file_out>"],50)
            print "SPEARMAN", i, spearman 

