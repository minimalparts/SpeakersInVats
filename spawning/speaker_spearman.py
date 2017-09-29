#speaker-spearman.py
#argv[1]: space dm file 1
#argv[2]: space dm file 2
#argv[3]: MEN file
#argv[4]: num dim for SVD
#EXAMPLE: python spearman.py wiki.S1.dm wiki.S2.dm MEN_dataset_natural_form_full 50
#-------
from composes.utils import io_utils
from composes.semantic_space.space import Space
from composes.utils import scoring_utils
from composes.similarity.cos import CosSimilarity
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.scaling.row_normalization import RowNormalization
import sys


#read in spaces
def read_space(space_file,dim):
    space = Space.build(data=space_file, cols="../spaces/wiki_all.cols", format='dm')
    #apply PPMI before calculating rho
    space = space.apply(PpmiWeighting())
    space = space.apply(RowNormalization())
    space = space.apply(Svd(dim))
    return space

def run_spearman(s1,s2,num_svd_dims):

    #number of dimensions for SVD
    dim = num_svd_dims

    space1 = read_space(s1,dim)
    space2 = read_space(s2,dim)

    #compute similarities of a list of word pairs
    fname = "MEN_dataset_natural_form_full"
    word_pairs = io_utils.read_tuple_list(fname, fields=[0,1])
    print len(word_pairs), "word pairs..."

    predicted1=[]
    predicted2=[]
    cos=0
    for wp in word_pairs:
        cos1=space1.get_sim(wp[0],wp[1], CosSimilarity())
        cos2=space2.get_sim(wp[0],wp[1], CosSimilarity())
        if cos1 != 0 and cos2 != 0:		#If word is not in speaker's vocabulary, it would be unfair to consider it for rho
            predicted1.append(cos1)
            predicted2.append(cos2)
        else:
            print wp[0],wp[1],cos1,cos2
 

    #compute correlations
    print len(predicted1),"pairs considered..."
    sp = scoring_utils.score(predicted1, predicted2, "spearman")
    print "Spearman",sp
    return sp
