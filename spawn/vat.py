"""Create speakers by adding noise to a reference matrix

Usage:
vat.py <file_in> <dimensionality> <num_speakers> --perturbation=<type>
vat.py --version

Options:
-h --help     Show this screen.
--version     Show version.
"""

import os
import evals
import sys
import random
import numpy as np
from utils import read_external_vectors,ppmi,normalise_l2,compute_PCA,print_matrix, print_vocab, print_dict
from docopt import docopt
import shutil


def mk_vat(args,vocab,speakers,vat_dir):
    num_files_in_dir = len([name for name in os.listdir(vat_dir)])
    dir_num = num_files_in_dir+1
    new_dir = "vat"+str(dir_num)
    os.makedirs(vat_dir+'/'+new_dir)

    for c in range(len(speakers)):
        print_matrix(speakers[c],vocab,vat_dir+'/'+new_dir+'/s'+str(c)+'.dm')

    print_vocab(vocab,vat_dir+'/'+new_dir+'/vocab.txt')
    print_dict(args,vat_dir+'/'+new_dir+'/settings.txt')
    shutil.make_archive(vat_dir+'/'+new_dir,'zip',vat_dir,new_dir)
    shutil.rmtree(vat_dir+'/'+new_dir)
    

def get_realistic_perturbation(original,frozen):
    new_vec = original.copy()
    perc = random.randint(95,105)
    add = np.random.choice([0,1],p=[0.99,0.01])
    for i in range(len(new_vec)):
        if i not in frozen:
            new_vec[i] = int(perc * original[i] / 100)
            new_vec[i] = max(0,new_vec[i]+add)
    return np.array(new_vec)


def reset_word(dm_mat,i,new_random):
    dm_mat[i] = new_random
    dm_mat = dm_mat.T
    dm_mat[i] = new_random
    dm_mat = dm_mat.T
    return dm_mat

def process_matrix(m,dim):
    m = ppmi(m)
    m = normalise_l2(m)
    m = compute_PCA(m,dim)
    return m

def make_speaker(m,vocab,dim):
    frozen= []
    ref_speaker = np.copy(m)
    test_speaker = np.copy(m)
    for i in range(test_speaker.shape[0]):
        original = test_speaker[i]
        new_random = get_realistic_perturbation(original,frozen)
        test_speaker = reset_word(test_speaker,i,new_random)
        frozen.append(i)
    
    ref_speaker_proc = process_matrix(ref_speaker,dim)
    test_speaker_proc = process_matrix(test_speaker,dim)

    #between_speakers = evals.RSA(ref_speaker_proc,test_speaker_proc)[0]
    sp1_men = evals.compute_men_spearman(ref_speaker_proc,vocab)[0]
    sp2_men = evals.compute_men_spearman(test_speaker_proc,vocab)[0]
    return test_speaker_proc, sp1_men, sp2_men


if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    m, vocab = read_external_vectors(args["<file_in>"])
    dimensionality = int(args["<dimensionality>"])

    vat_dir = "./allvats"
    if not os.path.exists(vat_dir):
        os.makedirs(vat_dir)

    speakers=[]
    all_sp1_men = []
    all_sp2_men = []



    for i in range(int(args["<num_speakers>"])):
        print("Making speaker",i,"...")
        ref_speaker = np.copy(m)
        test_speaker, sp1_men, sp2_men = make_speaker(ref_speaker,vocab,dimensionality)
        all_sp1_men.append(sp1_men)
        all_sp2_men.append(sp2_men)

        print("SPEARMAN MEN REF",sp1_men)
        print("SPEARMAN MEN SPAWNED",sp2_men)
        speakers.append(test_speaker)
        
    mk_vat(args,vocab,speakers,vat_dir)

    print("\n\nMEAN SPEARMAN MEN REF:",sum(all_sp1_men)/len(all_sp1_men))
    print("MEAN SPEARMAN MEN SPAWNED:",sum(all_sp2_men)/len(all_sp2_men))
