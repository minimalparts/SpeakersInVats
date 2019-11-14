"""Visualise all vats in a folder

Usage:
vat.py --dir=<dirname>  [--locus=<n>]
vat.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import sys
import shutil
import numpy as np
from docopt import docopt
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
    value = 0.0
    locus = None
    f = open(join(d,"settings.txt"))
    for l in f:
        l = l.rstrip('\n')
        print(l)
        if l[:4] == '--v ':
            value = l.split()[1]
        if l[:8] == '--locus ':
            locus = l.split()[1]
    return value, locus

def draw_matrices(matrices):
    cmap = get_cmap()
    color=0
    for m in matrices:
        color+=1
        plt.plot(m[:,0],m[:,1],'o',color=cmap(color))
    plt.show()

def draw_correlations(vatdir, rsa_to_control, corr_to_men, values, loci, locus=None):
    fig,ax = plt.subplots()
    cmap = get_cmap()
    color=0
    for i in range(len(rsa_to_control)):
        if locus and loci[i] != locus: #or float(values[i]) > 1.3:
            continue
        if loci[0] == None:
            color+=1
            plt.plot(rsa_to_control[i],corr_to_men[i],'o',color=cmap(color))
        else:
            plt.plot(rsa_to_control[i],corr_to_men[i],'o',color=cmap(int(loci[i])))
            
        plt.annotate(values[i], xy=(rsa_to_control[i][0], corr_to_men[i][0]), xytext=(0, 10), textcoords='offset points', color='black', size=10)
    ax.set_xlabel('RSA to control')
    ax.set_ylabel('Spearman to MEN')
    filename = vatdir+".summary."+locus+".png" if locus else vatdir+".summary.png"
    plt.savefig(join("./img",filename))
        

def get_speaker_data(vatdir):
    speakers = []
    rsa_to_control = []
    corr_to_men = []
    values = []
    loci = []
    speaker_files = [join(vatdir,f) for f in listdir(vatdir)]
    for sfile in speaker_files:
        print("Processing",sfile,"...")
    
        unpack(sfile,vatdir)
        vat = sfile.replace(".zip","")
        value,locus = read_params(vat)
        values.append(value)
        loci.append(locus)

        #speakers.append(np.load(join(vat,"rsa.2d.npy")))
        rsa_to_control.append(np.load(join(vat,"rsa.ref.test.npy")))
        corr_to_men.append(np.load(join(vat,"corr.men.npy")))
 
        shutil.rmtree(vat)
    return speakers, rsa_to_control, corr_to_men, values, loci


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    print(args)

    vatdir = args["--dir"]
    if args["--locus"]:
       locus = args["--locus"]
    else:
        locus = None
    speakers, rsa_to_control, corr_to_men, values, loci = get_speaker_data(vatdir)
    draw_correlations(vatdir, rsa_to_control, corr_to_men, values, loci, locus)

