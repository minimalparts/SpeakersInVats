import sys
import shutil
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import combinations
from os import listdir
from os.path import isfile, join
from utils import read_external_vectors, compute_PCA
from evals import RSA, compute_cosines

def get_cmap():
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.brg(vals))
    return cmap


def unpack(f,vatdir):
    shutil.unpack_archive(f, extract_dir=vatdir)


def read_params(d):
    f = open(join(d,"settings.txt"))
    for l in f:
        l = l.rstrip('\n')
        print(l)
        if l[:4] == '--v ':
            value = l.split()[1]
    return value

def draw_matrices(matrices):
    cmap = get_cmap()
    color=0
    for m in matrices:
        color+=1
        plt.plot(m[:,0],m[:,1],'o',color=cmap(color))
    plt.show()

def draw_correlations(rsa_to_control, corr_to_men, values):
    cmap = get_cmap()
    color=0
    for i in range(len(rsa_to_control)):
        color+=1
        plt.plot(rsa_to_control[i],corr_to_men[i],'o',color=cmap(color))
        plt.annotate(values[i], xy=(rsa_to_control[i][0], corr_to_men[i][0]), xytext=(0, 10), textcoords='offset points', color='black', size=10)
    plt.show()
        

def get_speaker_data(vatdir):
    speakers = []
    rsa_to_control = []
    corr_to_men = []
    values = []
    speaker_files = [join(vatdir,f) for f in listdir(vatdir)]
    for sfile in speaker_files:
        print("Processing",sfile,"...")
    
        unpack(sfile,vatdir)
        vat = sfile.replace(".zip","")
        values.append(read_params(vat))

        #speakers.append(np.load(join(vat,"rsa.2d.npy")))
        rsa_to_control.append(np.load(join(vat,"rsa.ref.test.npy")))
        corr_to_men.append(np.load(join(vat,"corr.men.npy")))
 
        shutil.rmtree(vat)
    return speakers, rsa_to_control, corr_to_men, values

vatdir = sys.argv[1]

speakers, rsa_to_control, corr_to_men, values = get_speaker_data(vatdir)
draw_correlations(rsa_to_control, corr_to_men, values)

