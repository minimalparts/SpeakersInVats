import sys
from utils import readDM, readCols, run_PCA, run_STNE
import matplotlib.pyplot as plt
import numpy as np

dm_mat, word_to_i, i_to_word = readDM(sys.argv[1])
cols=readCols(sys.argv[2])

'''For each dimension, get the word which gives the
min / max on that dimension. Those words will be on
the convex hull (but not only those).'''

col_mat = dm_mat.T

minimums = {}
maximums = []
c = 0
for row in col_mat[:1000]:
    print("ROW",cols[c])
    maximum = np.argmax(row)
    print("max:",i_to_word[maximum],np.max(row))
    maximums.append(i_to_word[maximum])
    minimum = np.argmin(row)
    minimum = np.argwhere(row == np.amin(row))
    print("min:",[i_to_word[m] for m in minimum.flatten().tolist()][:20])
    for m in minimum.flatten().tolist():
        word = i_to_word[m]
        if word in minimums:
            minimums[word]+=1
        else:
            minimums[word] = 1
    c+=1

print(minimums)
print(set(maximums))

#plt.scatter(dm_mat2[:,0],dm_mat2[:,1],marker='o')
#for i, x, y in zip(i_to_word.keys(), dm_mat2[:, 0], dm_mat2[:, 1]):
#    label = i_to_word[i]
#    plt.annotate(label,xy=(x, y), xytext=(-5, 5), textcoords='offset points')
#plt.show()
