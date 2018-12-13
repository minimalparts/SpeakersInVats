# Perturbation code

The code in this folder applies some perturbation to a reference matrix and computes the effect of that perturbation on a representational similarity dataset (MEN, [Bruni et al (2014)](https://staff.fnwi.uva.nl/e.bruni/MEN)).

## Run the code

To run the code, several hyperparameters must be provided. The first one is the type of perturbation to apply. The top of noise.py describes the various options currently available, reproduced below for convenience:

    Usage:
    noise.py <file_in> <file_out> random_reset (--dense|--sparse) --from=NUM --to=NUM <num_vec_perturbed>
    noise.py <file_in> <file_out> specific_reset (--dense|--sparse) <word>
    noise.py <file_in> <file_out> linear_reset --from=NUM --to=NUM <n> <num_vec_perturbed>
    noise.py <file_in> <file_out> exponential_reset --from=NUM --to=NUM <n> <num_vec_perturbed>
    noise.py <file_in> <file_out> shuffled_reset --from=NUM --to=NUM <num_vec_perturbed>
    noise.py <file_in> <file_out> make_speaker
    noise.py --version

    Options:
    -h --help     Show this screen.
    --version     Show version.

An example usage might be:

    python noise.py ../spaces/wiki.1M.dm tmp random_reset --dense --from=100 --to=200 5

This would call the code on the raw frequency matrix in *../spaces/wiki.1M.dm*, and output the modified vectors to *tmp*. The perturbation applied is *random_reset*, applied in a dense fashion (the zeros in the vector may be perturbed). 5 words are randomly chosen in the frequency range 100-200 (where 100 and 200 are the ranks of the words in the precompiled frequency list: so the 100th and 200th most frequent words).

The various options are described further here:

* random_reset: randomly perturbs the word's vector, either in a dense way (the vector is completely replaced) or in a sparse way (only the non-zero values are perturbed).
* specific_reset: like random_reset, but acts upon a particular word chosen by the user.
* linear_reset: simply multiplies the vectors by a real number *n*, it has the effect of changing the frequency of that word, but not its distribution.
* exponential_reset: applies exponential noise to the vectors, where *n* controls how much the distribution 'peaks' or 'flattens'.
* shuffled_reset: shuffles the elements of the vector, so that the word's overall frequency remains the same but its distribution changes.


## Output

For each modified vector, the system outputs some information exemplified in the following example:

    RANDOM WORD: under 63236
    ORIGINAL [  3.   1.  30.  10.   0.   2.  38.  23.   1.   3.] 63236.0
    1000 4373
    FO 63236 FN 4375.0
    NEW [ 7.  2.  5.  4.  3.  1.  3.  2.  8.  5.] 4375.0
    Running dense svd
    Running dense svd
    3000 word pairs...
    3000 pairs considered...
    Spearman 0.930905924767

The first line shows the word being perturbed (here, *under*) and its original frequency in the corpus (63236). The line showing 'FO' and 'FN' shows the difference in word frequency caused by the perturbation: here, *under* originally had a frequency of 63236, but it is now only 4375. Finally, the Spearman figure at the end show the correlation of the modified space with the original: in the above case, modifiying the word *under* with a random perturbation resulted in the Spearman rank to fall from an ideal 1 to 0.9309.
