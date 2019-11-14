import sys
import numpy as np
from utils import get_vocab_freqs, read_external_vectors, print_dict
import matplotlib.pyplot as plt

m,vocab = read_external_vectors(sys.argv[1])
freqs = get_vocab_freqs(m,vocab)
dfreqs = dict(zip(vocab,freqs))

for w in sorted(dfreqs, key=dfreqs.get, reverse=True):
    print(vocab.index(w),w,dfreqs[w])

mini = np.min(freqs)
maxi = np.max(freqs)

for i in range(10,100,10):
    perc = np.percentile(freqs,i)
    print(i,'%',perc)

bins = np.linspace(start = mini, stop = maxi, num = 50)
plt.hist(freqs, bins=bins)
#plt.show()
