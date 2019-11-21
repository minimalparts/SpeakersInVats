"""Analyse mapped terrain

Usage:
analyse_terrain.py --map=<file>
analyse_terrain.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import sys
import shutil
import numpy as np
from docopt import docopt
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_xy(terrain,v1,v2):
    X = []
    Y = []
    for entry in terrain:
        for k,v in entry.items():
            if k == v1:
                X.append(float(v))
            if k == v2:
                Y.append(float(v))
    return X,Y

def rmse_avgs(terrain):
    num_nns = 0
    avg_reds = {2:[],5:[],10:[],20:[],50:[]}
    avg_finals = {2:[],5:[],10:[],20:[],50:[]}
    for entry in terrain:
        num_nns = entry["NUM NN"]-1
        avg_reds[num_nns].append(float(entry["RMSE RED"]))
        avg_finals[num_nns].append(float(entry["RMSE FINAL"]))
    for n,l in avg_reds.items():
        avg_reds[n] = sum(l) / len(l)
    for n,l in avg_finals.items():
        avg_finals[n] = sum(l) / len(l)
    return avg_reds, avg_finals


def make_figure(X,Y):
    plt.plot(X, Y, 'o', label = 'xy', color='red')
    plt.show()

def read_map(mapfile):
    terrain = []
    d,locus,value="","",""
    mf = open(mapfile,'r')
    for l in mf:
        l = l.rstrip('\n')
        if "--dir':" in l:
            d = l[:-2].replace(" '--dir': '","")
        if "--locus':" in l:
            locus = l[-3:-2]
        if "--v':" in l:
            value = l[-5:-2]
        if l[0].isalpha():
           fields=l.split('\t')
           terrain_entry = {}
           for f in fields[1:]:
               f = f.split(':')
               terrain_entry[f[0]] = f[1].strip()
           terrain_entry["NUM NN"] = len(terrain_entry["NEIGH"].split())
           terrain_entry["RMSE RED"] = str(float(terrain_entry["RMSE ORIG"]) / float(terrain_entry["RMSE FINAL"]))
           #print(fields[0],terrain_entry["RMSE ORIG"],terrain_entry["RMSE FINAL"],terrain_entry["RMSE RED"],terrain_entry["DENSITY"])
           terrain.append(terrain_entry)
    return d, locus, value, terrain 


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')

    mapfile = args["--map"]
    d,locus,value,terrain = read_map(mapfile)

    avg_reds, avg_finals = rmse_avgs(terrain)
    
    print("AVG RMSE REDUCTIONS:")
    for k,v in avg_reds.items():
        print(k,v)

    print("\nAVG RMSE FINALS:")
    for k,v in avg_finals.items():
        print(k,v)

    X,Y = get_xy(terrain,"DENSITY","RMSE RED")
    print("\nDENSITY / RMSE REDUCTION:",spearmanr(X,Y),"\n")
    
    X,Y = get_xy(terrain,"ISOLATION","RMSE RED")
    print("ISOLATION / RMSE REDUCTION:",spearmanr(X,Y),"\n")

    X,Y = get_xy(terrain,"NUM NN","RMSE RED")
    print("NUM NN / RMSE REDUCTION:",spearmanr(X,Y),"\n")

    X,Y = get_xy(terrain,"FREQ","RMSE RED")
    print("FREQ / RMSE REDUCTION:",spearmanr(X,Y),"\n")

    X,Y = get_xy(terrain,"DENSITY","RMSE FINAL")
    print("DENSITY / RMSE FINAL:",spearmanr(X,Y),"\n")
    
    X,Y = get_xy(terrain,"ISOLATION","RMSE FINAL")
    print("ISOLATION / RMSE FINAL:",spearmanr(X,Y),"\n")

    X,Y = get_xy(terrain,"NUM NN","RMSE FINAL")
    print("NUM NN / RMSE FINAL:",spearmanr(X,Y),"\n")

    X,Y = get_xy(terrain,"FREQ","RMSE FINAL")
    print("FREQ / RMSE FINAL:",spearmanr(X,Y),"\n")

        
    #make_figure(X,Y)
