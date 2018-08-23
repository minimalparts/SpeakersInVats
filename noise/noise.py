"""Create speakers by adding noise to a reference matrix

Usage:
noise.py <file_in> <file_out> random_reset (--dense|--sparse) --from=NUM --to=NUM <num_vec_perturbed>
noise.py <file_in> <file_out> specific_reset (--dense|--sparse) <word>
noise.py <file_in> <file_out> linear_reset --from=NUM --to=NUM <n> <num_vec_perturbed>
noise.py <file_in> <file_out> exponential_reset --from=NUM --to=NUM <n> <num_vec_perturbed>
noise.py <file_in> <file_out> shuffled_reset --from=NUM --to=NUM <num_vec_perturbed>
noise.py <file_in> <file_out> make_speaker
noise.py --version

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

fixed = []		#rows/columns already perturbed

def get_random_word_in_range(word_freqs,begin,end):
    sliced = dict(sorted(word_freqs.iteritems(), key=operator.itemgetter(1), reverse=True)[begin:end])
    word = random.choice(sliced.keys())
    while word_to_i[word] in fixed:
        word = random.choice(sliced.keys())
    print "\nRANDOM WORD:",word, word_freqs[word]
    fixed.append(word_to_i[word])
    return word


def get_dense_random(original, word_freqs):
    print "ORIGINAL",original[:10], original.sum()
    random_freq = utils.get_rand_freq(word_freqs)
    freq_remainder = random_freq - sum(original[i] for i in fixed[:-1])	#Last element in fixed is current word
    while freq_remainder < 0:			#Rejection sampling: there should be a remainder to play with!!
        random_freq = utils.get_rand_freq(word_freqs)
        freq_remainder = random_freq - sum(original[i] for i in fixed[:-1])	#Last element in fixed is current word
    random_remainder = utils.get_rand_int_vec(len(original)-len(fixed[:-1]),freq_remainder)
    new_random = original.copy()
    c = 0
    for i in range(len(new_random)):
        if i not in fixed[:-1]:
            new_random[i] = random_remainder[c]
            c+=1
    return np.array(new_random)


def get_sparse_random(original, word_freqs):
    #print "ORIGINAL",original[:10], original.sum()
    zeros = utils.get_zero_positions(original)
    fixed_zeros = list(set(zeros+fixed[:-1]))		
    random_freq = utils.get_rand_freq(word_freqs)
    freq_remainder = random_freq - sum(original[i] for i in fixed_zeros[:-1])	#Last element in fixed is current word
    while freq_remainder < 0:			#Rejection sampling: there should be a remainder to play with!!
        random_freq = utils.get_rand_freq(word_freqs)
        freq_remainder = random_freq - sum(original[i] for i in fixed_zeros[:-1])	#Last element in fixed is current word
    random_remainder = utils.get_rand_int_vec(len(original)-len(fixed_zeros),freq_remainder)
    new_random = original.copy()
    c = 0
    for i in range(len(new_random)):
        if i not in fixed_zeros:
            new_random[i] = random_remainder[c]
            c+=1
    return np.array(new_random)

def get_shuffled_perturbation(original):
    print "ORIGINAL",original[:10], original.sum()
    zeros = utils.get_zero_positions(original)
    fixed_zeros = list(set(zeros+fixed[:-1]))
    to_shuffle = []
    for i in range(len(original)):
        if i not in fixed_zeros:
            to_shuffle.append(original[i])
    shuffled_remainder = np.random.permutation(to_shuffle)
    new_shuffled = original.copy()
    c = 0
    for i in range(len(new_shuffled)):
        if i not in fixed_zeros:
            new_shuffled[i] = shuffled_remainder[c]
            c+=1
    return np.array(new_shuffled)

def get_minimal_perturbation(original):
    print "ORIGINAL",original[:10], original.sum()
    new_linear = original.copy()
    perc = random.randint(80,120)
    #add = random.randint(-5,5)
    c = 0
    for i in range(len(new_linear)):
        if i not in fixed[:-1]:
            new_linear[i] = int(perc * original[i] / 100)
            #new_linear[i] = max(0,original[i]+add)
            c+=1
    return np.array(new_linear)


def get_realistic_perturbation(original):
    print "ORIGINAL",original[:10], original.sum()
    new_vec = original.copy()
    perc = random.randint(95,105)
    add = np.random.choice([0,1],p=[0.99,0.01])
    c = 0
    for i in range(len(new_vec)):
        if i not in fixed[:-1]:
            new_vec[i] = int(perc * original[i] / 100)
            new_vec[i] = max(0,new_vec[i]+add)
            c+=1
    return np.array(new_vec)


def get_linear_perturbation(original,n):
    new_linear = original.copy()
    c = 0
    for i in range(len(new_linear)):
        if i not in fixed[:-1]:
            new_linear[i] = int(float(n)*float(original[i]))
            c+=1
    return np.array(new_linear)


def get_exponential_perturbation(original,n):
    new_exponential = original.copy()
    c = 0
    for i in range(len(new_exponential)):
        if i not in fixed[:-1]:
            new_exponential[i] = int(math.pow(original[i],n)*10)
            c+=1
    return np.array(new_exponential)


def reset_word(dm_mat,word,new_random):
    print "NEW",new_random[:10], new_random.sum()
    dm_mat[word_to_i[word]] = new_random
    dm_mat = dm_mat.T
    dm_mat[word_to_i[word]] = new_random
    dm_mat = dm_mat.T
    return dm_mat


def specific_reset(dm_mat,dense,word,word_freqs):
    freq_diffs=[]
    original = dm_mat[word_to_i[word]]
    if dense:
        new_random = get_dense_random(original, word_freqs)
        freq_diffs.append(utils.compute_freq_diff(word_freqs[word],new_random))
    else:
        new_random = get_sparse_random(original, word_freqs)
        freq_diffs.append(utils.compute_freq_diff(word_freqs[word],new_random))
    dm_mat = reset_word(dm_mat,word,new_random)
    #print dm_mat
    return dm_mat, freq_diffs

def random_reset(dm_mat,dense,fromint,toint,num_vec, word_freqs):
    freq_diffs=[]
    for i in range(num_vec):
        w = get_random_word_in_range(word_freqs,fromint,toint)
        original = dm_mat[word_to_i[w]]
        if dense:
            new_random = get_dense_random(original, word_freqs)
            freq_diffs.append(utils.compute_freq_diff(word_freqs[w],new_random))
        else:
            new_random = get_sparse_random(original, word_freqs)
            freq_diffs.append(utils.compute_freq_diff(word_freqs[w],new_random))
        dm_mat = reset_word(dm_mat,w,new_random)
        #print dm_mat
    return dm_mat, freq_diffs

def shuffled_reset(dm_mat, fromint, toint, num_vec, word_freqs):
    freq_diffs=[]
    for i in range(num_vec):
        w = get_random_word_in_range(word_freqs,fromint,toint)
        original = dm_mat[word_to_i[w]]
        new_shuffled = get_shuffled_perturbation(original)
        freq_diffs.append(utils.compute_freq_diff(word_freqs[w],new_shuffled))
        dm_mat = reset_word(dm_mat,w,new_shuffled)
        #print dm_mat
    return dm_mat, freq_diffs

def linear_reset(dm_mat, fromint, toint, num_vec, n, word_freqs):
    freq_diffs=[]
    for i in range(num_vec):
        w = get_random_word_in_range(word_freqs,fromint,toint)
        original = dm_mat[word_to_i[w]]
        new_linear = get_linear_perturbation(original,n)
        freq_diffs.append(utils.compute_freq_diff(word_freqs[w],new_linear))
        dm_mat = reset_word(dm_mat,w,new_linear)
        #print dm_mat
    return dm_mat, freq_diffs

def exponential_reset(dm_mat, fromint, toint, num_vec, n, word_freqs):
    freq_diffs=[]
    for i in range(num_vec):
        w = get_random_word_in_range(word_freqs,fromint,toint)
        original = dm_mat[word_to_i[w]]
        new_exponential = get_exponential_perturbation(original,n)
        freq_diffs.append(utils.compute_freq_diff(word_freqs[w],new_exponential))
        dm_mat = reset_word(dm_mat,w,new_exponential)
        #print dm_mat
    return dm_mat, freq_diffs


def make_speaker(dm_mat):
    #sliced = dict(sorted(word_freqs.iteritems(), key=operator.itemgetter(1), reverse=True)[50:])
    sliced = dict(sorted(word_freqs.iteritems(), key=operator.itemgetter(1), reverse=True))
    c=0
    for w,f in sliced.items():
        print "CHANGING",w
        original = dm_mat[word_to_i[w]]
        new_random = get_realistic_perturbation(original)
        dm_mat = reset_word(dm_mat,w,new_random)
        c+=1
    print "CHANGED",c,"ITEMS"
    return dm_mat


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    #print args

    dm_mat_ref, word_to_i, i_to_word, word_freqs, max_freq = utils.readDM(args["<file_in>"])
    print "MAX FREQ:", max_freq
    means = []
    stds = []

    if args["make_speaker"]:
        dm_mat_bck = np.copy(dm_mat_ref)
        test_speaker = make_speaker(dm_mat_ref)
        utils.printDM(test_speaker,args["<file_out>"], i_to_word)
        spearman=speaker_spearman.run_spearman(args["<file_in>"],args["<file_out>"],50)
        print "SPEARMAN", spearman 

    else:
        spearmans=[]
        for i in range(10):
            dm_mat_bck = np.copy(dm_mat_ref)
            #print dm_mat_ref
            if args["random_reset"]:
                test_speaker, diffs = random_reset(dm_mat_ref,args["--dense"],int(args["--from"]),int(args["--to"]),int(args["<num_vec_perturbed>"]), word_freqs)
            if args["shuffled_reset"]:
                test_speaker, diffs = shuffled_reset(dm_mat_ref,int(args["--from"]),int(args["--to"]),int(args["<num_vec_perturbed>"]), word_freqs)
            if args["linear_reset"]:
                test_speaker, diffs = linear_reset(dm_mat_ref,int(args["--from"]),int(args["--to"]),int(args["<num_vec_perturbed>"]), float(args["<n>"]), word_freqs)
            if args["exponential_reset"]:
                test_speaker, diffs = exponential_reset(dm_mat_ref,int(args["--from"]),int(args["--to"]),int(args["<num_vec_perturbed>"]), float(args["<n>"]), word_freqs)
            if args["specific_reset"]:
                test_speaker, diffs = specific_reset(dm_mat_ref, args["--dense"], args["<word>"], word_freqs)
            means.append(np.mean(np.array(diffs)))
            stds.append(np.std(np.array(diffs)))
            utils.printDM(test_speaker,args["<file_out>"], i_to_word)
            spearman=speaker_spearman.run_spearman(args["<file_in>"],args["<file_out>"],50)
            spearmans.append(spearman)
            dm_mat_ref = dm_mat_bck	#reset control matrix
            fixed = []			#reset fixed
        
        print "AVG SPEARMAN",np.mean(np.array(spearmans))
        print "AVG SPEARMAN STD",np.std(np.array(spearmans))
        print "AVG FREQ DIFF MEAN",np.mean(np.array(means))
        print "AVG FREQ  DIFF STD",np.std(np.array(stds))

