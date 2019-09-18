from scipy.stats import spearmanr
from scipy.spatial import distance

def compute_spearman(m,vocab):
    system = []
    gold = []

    f = open("data/MEN_dataset_natural_form_full",'r')
    for l in f:
        fields = l.rstrip('\n').split()
        w1 = fields[0]
        w2 = fields[1]
        score = float(fields[2])
        if w1 in vocab and w2 in vocab:
            cos = 1 - distance.cosine(m[vocab.index(w1)],m[vocab.index(w2)])
            system.append(cos)
            gold.append(score)
            #print(w1,w2,cos,score)
    f.close()

    print(len(gold),"pairs considered...")
    return spearmanr(system,gold)

