import math
import random
import numpy as np

def get_zero_perturbation(original,param,frozen):
    new_vec = original.copy()
    add = np.random.choice([0,1],p=[1-param,param])
    for i in range(len(new_vec)):
        if i not in frozen:
            new_vec[i] = max(0,new_vec[i]+add)
    return np.array(new_vec)

def get_linear_perturbation(original,param,frozen):
    new_linear = original.copy()
    c = 0
    for i in range(len(new_linear)):
        if i not in frozen:
            new_linear[i] = int(float(n)*float(original[i]))
            c+=1
    return np.array(new_linear)

def get_exponential_perturbation(original,param,frozen):
    new_exponential = original.copy()
    for i in range(len(new_exponential)):
        if i not in frozen[:-1]:
            new_exponential[i] = int(math.pow(original[i],param)*10)
    return np.array(new_exponential)

def get_random_perturbation(original,param,frozen,zp=False):
    if zp:
        frozen = list(set(zeros+frozen))		
    random_freq = random.randint(0,param)
    freq_remainder = random_freq - sum(original[i] for i in frozen)
    while freq_remainder < 0:			#Rejection sampling: there should be a remainder to play with!!
        random_freq = utils.get_rand_freq(word_freqs)
        freq_remainder = random_freq - sum(original[i] for i in frozen)
    random_remainder = utils.get_rand_int_vec(len(original)-len(frozen),freq_remainder)
    new_vec = original.copy()
    c = 0
    for i in range(len(new_vec)):
        if i not in frozen:
            new_vec[i] = random_remainder[c]
            c+=1
    return np.array(new_vec)

def get_shuffled_perturbation(original):
    #print "ORIGINAL",original[:10], original.sum()
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

