import sys
from utils import readDM, run_PCA, run_STNE, run_MDS
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
    dim1 = sys.argv[2]
    dim2 = sys.argv[3]


dm_mat, word_to_i, i_to_word = readDM(sys.argv[1])
#dm_mat2 = run_PCA(dm_mat)
#dm_mat2 = run_STNE(dm_mat)
#dm_mat2 = run_MDS(dm_mat)
dm_mat2 = dm_mat[:,[word_to_i[dim1],word_to_i[dim2]]]
hull = ConvexHull(dm_mat2)

for vertice in hull.vertices:
    print(vertice,i_to_word[vertice],hull.points[vertice])
#dm_mat2 = run_MDS(dm_mat)
plt.scatter(dm_mat2[:,0],dm_mat2[:,1],marker='o')
for simplex in hull.simplices:
    plt.plot(dm_mat2[simplex,0], dm_mat2[simplex,1], 'k-')
for i, x, y in zip(i_to_word.keys(), dm_mat2[:, 0], dm_mat2[:, 1]):
    label = i_to_word[i]
    plt.annotate(label,xy=(x, y), xytext=(-5, 5), textcoords='offset points')
plt.show()
#plt.savefig("convex-hull.png")
