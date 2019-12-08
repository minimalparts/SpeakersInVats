import math
import random
import numpy as np
from utils.utils import get_zero_positions, get_rand_freq, get_rand_ints_with_sum

def get_zero_perturbation(original,param,frozen):
    new_vec = original.copy()
    for i in range(len(new_vec)):
        if i not in frozen and new_vec[i] == 0:
            add = np.random.choice([0,1],p=[1-param,param])
            new_vec[i] = max(0,new_vec[i]+add)
    return np.array(new_vec)

def get_linear_perturbation(original,param,frozen):
    perc = random.randint(100-100*param,100+100*param)
    new_vec = original.copy()
    c = 0
    for i in range(len(new_vec)):
        if i not in frozen:
            new_vec[i] = int(perc * original[i] / 100)
            c+=1
    return np.array(new_vec)

def get_exponential_perturbation(original,param,frozen):
    new_exponential = original.copy()
    for i in range(len(new_exponential)):
        if i not in frozen:
            new_exponential[i] = int(math.pow(original[i],param))
    return np.array(new_exponential)

def get_shuffled_perturbation(original, reference, frozen, zp=False):
    if zp:
        zeros = get_zero_positions(reference)
        frozen = list(set(zeros+frozen))
    to_shuffle = []
    for i in range(len(original)):
        if i not in frozen:
            to_shuffle.append(original[i])
    shuffled_remainder = np.random.permutation(to_shuffle)
    new_shuffled = original.copy()
    c = 0
    for i in range(len(new_shuffled)):
        if i not in frozen:
            new_shuffled[i] = shuffled_remainder[c]
            c+=1
    return np.array(new_shuffled)

