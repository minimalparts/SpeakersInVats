import sys
import shutil
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from utils import read_external_vectors, read_params, read_ranked_freqs

def get_cmap():
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.brg(vals))
    return cmap


def unpack(f,vatdir):
    shutil.unpack_archive(f, extract_dir=vatdir)


def draw_zipf(ranked_logs, values):
    cmap = get_cmap()
    log_ranks = ranked_logs[0][0]
    log_freqs = ranked_logs[0][1]
    plt.plot(log_ranks,log_freqs, 'ro-', markersize=5, color='black')
    plt.annotate('CONTROL', xy=(log_ranks[0], log_freqs[0]), xytext=(-5, 10), textcoords='offset points', color='black', size=10)
    color=0
    for i in range(1,len(ranked_logs)):
        color+=1
        log_ranks = ranked_logs[i][0]
        log_freqs = ranked_logs[i][1]
        plt.plot(log_ranks,log_freqs, 'ro-', markersize=1, color=cmap(color))
        plt.annotate(values[i], xy=(log_ranks[50], log_freqs[50]), xytext=(-5, 10), textcoords='offset points', color='black', size=10)
    plt.show()

def draw_correlations(rsa_to_control, corr_to_men, values):
    cmap = get_cmap()
    color=0
    for i in range(len(rsa_to_control)):
        color+=1
        plt.plot(rsa_to_control[i],corr_to_men[i],'o',color=cmap(color))
        plt.annotate(values[i], xy=(rsa_to_control[i][0], corr_to_men[i][0]), xytext=(-5, 10), textcoords='offset points', color='black', size=10)
    plt.show()
       

def get_speaker_data(vatdir):
    speakers = []
    ranked_logs = []
    speaker_files = [join(vatdir,f) for f in listdir(vatdir)]
    values = []
    for sfile in speaker_files:
        print("Processing",sfile,"...")
    
        unpack(sfile,vatdir)
        vat = sfile.replace(".zip","")
        params = read_params(join(vat,'settings.txt'))
        if ranked_logs == []:
            ranked_logs.append(read_ranked_freqs(join(vat,'ranked_vocab.txt')))
            values.append("REF")
        ranked_logs.append(read_ranked_freqs(join(vat,'s0.ranked_vocab.txt')))
        values.append(params["--v"])
        shutil.rmtree(vat)
    return ranked_logs, values

vatdir = sys.argv[1]

ranked_logs,values = get_speaker_data(vatdir)
draw_zipf(ranked_logs, values)

