# Semantic chaos

This is the repo accompanying the paper *Semantic chaos: Modelling the Effect of Creativity
and Individuality on Lexical Alignment* (in prep).

## Data 

The *data* directory contains a **control semantic space** created from a Wikipedia snapshot from November 2018, as well as a version of the MEN dataset ([Bruni et al, 2014](https://staff.fnwi.uva.nl/e.bruni/MEN)).

The semantic space was generated using a word window of ±2 words on
either side of the target. I was looking to produce a square matrix that combines a reasonable
size (to allow for fast experimentation) and good coverage on the MEN test set. After sampling a
number of spaces, I selected one which simply uses as row and column labels the vocabulary of
the MEN dataset, so that we end up with a matrix of dimension 750 × 750. When applying PPMI
weighting and dimensionality-reduction to this matrix (using Principal Component Analysis with
40 components), we get a correlation ρ = 0.67 with the average MEN ratings.



## Perturbing the control space

To run spawn new speakers from the control space, use one of the following:

    vat.py --in=<file> --dim=<n> --num_speakers=<n> --perturbation=<type> --v=<param_value> [--dir=<dirname>]
    vat.py --in=<file> --dim=<n> --num_speakers=<n> --perturbation=<type> --v=<param_value> --locus=<n> [--dir=<dirname>]
    vat.py --in=<file> --dim=<n> --num_speakers=<n> --perturbation=<type> --v=<param_value> --num=<n> --locus=<n> [--dir=<dirname>]

When running with the --locus flag, the perturbation will only be applied to a certain frequency range in the vocabulary. In addition, when running with --num, only $n$ randomly selected words receive perturbation in the given frequency range.


Four different pertubation types can be applied to the control space, all defined in utils/perturbations.py:

* **Unattested:** run with flag --perturbation=zeros. This creates unattested word combinations in the co-occurrence matrix by simply flipping 0 counts into 1s. Specifically, for a particular word vector *v*, we consider each 0 component in turn and flip it to 1 with some probability *p*.

* **Collocations:** run with flag --perturbation=exp. By increasing / decreasing a vector component as an exponential function of its value, we mimic an increase / decrease in the use of a certain collocation. For each component *v_k* in *v*, we set  *v_k = v_k^e*. For values of *e* above 1, this has the effect to amplify the shape of the existing distribution and simulates the case where the test speaker uses strong collocates of the word much more frequently than the control. Conversely, values under 1 'flatten' the distribution and erase the collocations.

* **Frequencies:** run with flag --perturbation=linear. Natural languages are known to have vocabularies that follow a Zipfian curve, with a few words being extremely frequent and many being extremely infrequent. We can modify the general Zipfian distribution by applying linear perturbations to the rows of the matrix: we increase / decrease each non-frozen vector component as a linear function of its value. That is, for each component *v_k* in *v*, we set  *v_k = xv_k*. This simulates a case where the word is much more/less frequent for the test speaker than for the control. Whilst this transformation does not have an effect on cosine distance in the count matrix, it does result in differences in the weighted, dimensionality-reduced spaces used to compute correlation between speakers.

* **Shuffling:** run with flag --perturbation=shuffle. A control condition showing the amount of alignment left in two radically different speakers. To implement this, we shuffle the components of the vector under two conditions: one where the sparsity of the vector is preserved (i.e. zero components are left untouched), and one where shuffling affects all components (thus producing unattested word combinations).  


An example usage is:

   python3 vat.py --in=data/enwiki-20181120.ss.toks.mincount-100.win-2.MEN.txt --dim=40 --num_speaker=10 --perturbation=exp --v=0.2 

The output will be something like:

    Making speaker 0 ...
    Percentile: None
    RSA REF TEST 0.4639118170923366
    SPEARMAN MEN REF 0.6716123142755714
    SPEARMAN MEN SPAWNED 0.6155885995528726
    
    Making speaker 1 ...
    Percentile: None
    RSA REF TEST 0.4639118170923366
    SPEARMAN MEN REF 0.6716123142755714
    SPEARMAN MEN SPAWNED 0.6155885995528726

