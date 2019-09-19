import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from os import listdir
from os.path import isfile, join
from utils import read_external_vectors, compute_PCA
from evals import RSA, compute_cosines


def unpack(f):
    shutil.unpack_archive(f, extract_dir="./allvats")


def return_pairs(l):
    return list(combinations(l,2))


def make_figure(m):
    plt.plot(m[:, 0], m[:, 1], 'o', label = 'data')
    plt.show()

unpack(sys.argv[1])
allvats = sys.argv[1].replace(".zip","")

speaker_files = [f for f in listdir(allvats) if ".dm" in join(allvats, f)]
print("Found",len(speaker_files),"speaker files...")

speakers = []

for s in speaker_files:
    m, vocab = read_external_vectors(join(allvats,s))
    m_cos = compute_cosines(m)
    speakers.append(m_cos)

rsa_matrix = np.zeros((len(speakers),len(speakers)))

for i in range(len(speakers)):
    for j in range(len(speakers)):
        rsa_matrix[i][j] = RSA(speakers[i],speakers[j])[0]

print(rsa_matrix)
red_rsa_matrix = compute_PCA(rsa_matrix,2)
print(red_rsa_matrix)
make_figure(red_rsa_matrix)

shutil.rmtree(allvats)
