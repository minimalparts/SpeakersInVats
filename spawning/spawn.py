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
import random
import numpy as np
import speaker_spearman, utils
from docopt import docopt


def sample_matrix_random(dm_mat_file):
    dm_mat, word_to_i, i_to_word, word_freqs, max_freq = utils.readDM(dm_mat_file)
    new_matrix = dm_mat.copy()
    for i in range(len(dm_mat[0])):
        for j in range(i):
            new_matrix[i,j] = np.random.choice([0,random.randint(0,dm_mat[i,j])], p=[9.9/10, 0.1/10])
            new_matrix[j,i] = new_matrix[i,j]
    return new_matrix, i_to_word

def make_speaker():
    spawned_speaker, i_to_word = sample_matrix_random("../spaces/speakers/speaker1.dm")
    for i in range(2,11):
        speaker_file = "../spaces/speakers/speaker"+str(i)+".dm"
        new_matrix, i_to_word = sample_matrix_random(speaker_file)
        spawned_speaker+=new_matrix
    return spawned_speaker, i_to_word



if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, spawn 0.1')
    print args


    if args["spawn"]:
        test_speaker, i_to_word = make_speaker()
        utils.printDM(test_speaker,args["<file_out>"], i_to_word)
        spearman=speaker_spearman.run_spearman("../spaces/speakers/speaker1.dm",args["<file_out>"],50)
        print "SPEARMAN", spearman 

