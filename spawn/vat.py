"""Create speakers by adding noise to a reference matrix

Usage:
vat.py --in=<file> --dim=<n> --num_speakers=<n> --perturbation=<type> --v=<param_value>
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
from os.path import join
import perturbations
from utils import is_number,read_external_vectors,ppmi,normalise_l2,compute_PCA,print_matrix, print_vocab, print_dict, print_list, print_list_pair,get_rand_freq, get_vocab_ranked_logs
from docopt import docopt
import shutil

def save_ref_to_test(vat,m):
    print("Saving to ",join(vat,'rsa.ref.test.npy'))
    np.save(join(vat,'rsa.ref.test'),m)

def save_corr_to_men(vat,m):
    print("Saving to ",join(vat,'corr.men.npy'))
    np.save(join(vat,'corr.men'),m)

def mk_vat(args,vocab,speakers,all_ref_to_test,all_test_men,ranked_logs,vat_dir):
    num_files_in_dir = len([name for name in os.listdir(vat_dir)])
    dir_num = num_files_in_dir+1
    new_dir = "vat"+str(dir_num)
    os.makedirs(vat_dir+'/'+new_dir)

    print_list_pair(ranked_logs[0][0],ranked_logs[0][1],vat_dir+'/'+new_dir+'/ranked_vocab.txt')
    for c in range(len(speakers)):
        print_matrix(speakers[c],vocab,vat_dir+'/'+new_dir+'/s'+str(c)+'.dm')
        log_ranks = ranked_logs[c+1][0]
        log_freqs = ranked_logs[c+1][1]
        print_list_pair(log_ranks,log_freqs,vat_dir+'/'+new_dir+'/s'+str(c)+'.ranked_vocab.txt')

    print_vocab(vocab,vat_dir+'/'+new_dir+'/vocab.txt')
    print_dict(args,vat_dir+'/'+new_dir+'/settings.txt')
    save_ref_to_test(join(vat_dir,new_dir),all_ref_to_test)
    save_corr_to_men(join(vat_dir,new_dir),all_test_men)
    shutil.make_archive(vat_dir+'/'+new_dir,'zip',vat_dir,new_dir)
    shutil.rmtree(vat_dir+'/'+new_dir)
    
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

def make_speaker(m,vocab,dim,pert,param):
    frozen= []
    ref_speaker = np.copy(m)
    test_speaker = np.copy(m)
    indices = list(range(test_speaker.shape[0]))
    random.shuffle(indices)
    for i in indices:
        original = test_speaker[i]
        new_random = original
        if pert == "zeros":
            new_random = perturbations.get_zero_perturbation(original,param,frozen)
        if pert == "exp":
            new_random = perturbations.get_exponential_perturbation(original,param,frozen)
        if pert == "linear":
            new_random = perturbations.get_linear_perturbation(original,param,frozen)
        if pert == "shuffle":
            zp = True if param == 'zp' else False
            new_random = perturbations.get_shuffled_perturbation(original,ref_speaker[i],frozen,zp) #Also passing ref speaker for original zeros
        test_speaker = reset_word(test_speaker,i,new_random)
        frozen.append(i)

    ranked_logs = get_vocab_ranked_logs(test_speaker,vocab)
    ref_speaker_proc = process_matrix(ref_speaker,dim)
    test_speaker_proc = process_matrix(test_speaker,dim)

    ref_to_test = evals.RSA(evals.compute_cosines(ref_speaker_proc),evals.compute_cosines(test_speaker_proc))[0]
    sp1_men = evals.compute_men_spearman(ref_speaker_proc,vocab)[0]
    sp2_men = evals.compute_men_spearman(test_speaker_proc,vocab)[0]
    return test_speaker_proc, ref_to_test, sp1_men, sp2_men, ranked_logs




if __name__=="__main__":
    args = docopt(__doc__, version='Speakers in vats, noise 0.1')
    print(args)
    m, vocab = read_external_vectors(args["--in"])
    dimensionality = int(args["--dim"])
    pert = args["--perturbation"]
    if is_number(args["--v"]):
        param = float(args["--v"])
    else:
        param = args["--v"]

    vat_dir = "./allvats"
    if not os.path.exists(vat_dir):
        os.makedirs(vat_dir)

    speakers=[]
    all_ref_to_test = []
    all_sp1_men = []
    all_sp2_men = []
    ranked_logs = [get_vocab_ranked_logs(m,vocab)]


    for i in range(int(args["--num_speakers"])):
        print("Making speaker",i,"...")
        ref_speaker = np.copy(m)
        test_speaker, ref_to_test, sp1_men, sp2_men, ranked = make_speaker(ref_speaker,vocab,dimensionality,pert,param)
        all_ref_to_test.append(ref_to_test)
        all_sp1_men.append(sp1_men)
        all_sp2_men.append(sp2_men)
        ranked_logs.append(ranked)

        print("RSA REF TEST",ref_to_test)
        print("SPEARMAN MEN REF",sp1_men)
        print("SPEARMAN MEN SPAWNED",sp2_men)
        speakers.append(test_speaker)
        
    mk_vat(args,vocab,speakers,all_ref_to_test,all_sp2_men,ranked_logs,vat_dir)


    print("\n\nMEAN RSA REF TEST:",sum(all_ref_to_test)/len(all_ref_to_test))
    print("\n\nMEAN SPEARMAN MEN REF:",sum(all_sp1_men)/len(all_sp1_men))
    print("MEAN SPEARMAN MEN SPAWNED:",sum(all_sp2_men)/len(all_sp2_men))
