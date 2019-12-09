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


## Visualisation

It is possible to visualise the effect of a particular perturbation for different parameter values. The following will visualise all spawned speakers from the collocation perturbation, applied to the entire vocabulary (assuming that the perturbation has been run and its output saved in the *vats* directory under the name *global-exp-arpack*.)

    python3 visualise_summary.py --dir=vats/global-exp-arpack/

The same can be run to investigate perturbations at a particular decile of the data. For instance, the following will visualise collocation perturbation applied to the lowest 10% of the data (assuming that the perturbation has been run on separate deciles and the output saved in the *vats/local-exp-arpack* directory.)

    python3 visualise_summary.py --dir=vats/local-exp-arpack/ --locus=0


## Modelling perturbations

We can now see to what extent local areas of the space can be modelled using rotations and scaling. The following takes the control space and the directory containing the results of a given perturbation, as generated by *vat.py*. Given a particular parameter value *--v* and a locus of perturbation *--locus*, it retrieves the correct vat from the directory and attempts to model the transformation from control to perturbed space for a given number of neighbourhoods (*--num_words* neighbourhoods with *--nns* nearest neighbours each). The transformation involves finding the best rotation and scaling on centered spaces.

    python3 model_perturbations.py --control=data/enwiki-20181120.ss.toks.mincount-100.win-2.MEN.txt --dir=vats/local-exp-arpack/ --v=0.5 --locus=9 --num_words=2 --nns=4

The output of the above might be

    TEST WORDS: ['work', 'man'] 

    NEIGHBOURHOOD: ['work', 'art', 'graphic', 'abstract', 'design']
    Processing vats/local-exp-arpack/vat50.zip ...
    ROTATING...
    SCALING... (FACTOR: 1.38 )
    NEIGHBOURHOOD: ['man', 'head', 'body', 'left', 'person']
    Processing vats/local-exp-arpack/vat50.zip ...
    ROTATING...
    SCALING... (FACTOR: 1.22 )
    AVG RMSE, CONTROL - PERTURBED: 0.07366644944833167
    AVG RMSE, ROTATED - PERTURBED: 0.030520423738435453
    AVG RMSE, ROTATED+SCALED - PERTURBED: 0.02817152011726653

That is, *work* and *man* are chosen as the two words to test. Their neighbourhoods in the control space are printed out. Rotation and scaling is applied. In the end, the average RMSE is given between a) the control and perturbed space; b) the rotated control space and the perturbed space; c) the rotated and scaled control space and the perturbed space. We see that the RMSE decreases, giving a measure of the extent to which rotation and scaling model the effect of the given perturbation.
