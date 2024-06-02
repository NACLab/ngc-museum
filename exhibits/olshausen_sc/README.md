# Sparse Coding (Olshausen &amp; Field, 1996)

<b>Version</b>: ngclearn>=1.1.beta1, ngcsimlib>=0.3.beta1

This exhibit contains an implementation of the sparse coding models proposed
and studied in:

Olshausen, B., Field, D. Emergence of simple-cell receptive field properties
by learning a sparse code for natural images. Nature 381, 607–609 (1996).

and the model proposed in: 

Daubechies, Ingrid, Michel Defrise, and Christine De Mol. "An iterative
thresholding algorithm for linear inverse problems with a sparsity constraint."
Communications on Pure and Applied Mathematics: A Journal Issued by the
Courant Institute of Mathematical Sciences 57.11 (2004): 1413-1457.

## Running the Model's Simulation

To train this implementation of sparse coding, simply the run convenience 
Bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for the provided natural scenes 
dataset. Alternatively, you can directly run the Python training script:

```console
$ python train_patch_sc.py --dataX="/path/to/train_patterns.npy" \
                           --model_type="<model_choice>" --n_iter=200 \
                           --verbosity=0
```

Note that you can point the training script(s) to other datasets besides the
default natural scenes dataset (as in the Bash script), just ensure that the
target for `dataX` is a numpy array of shape 
`(Number data points x Pattern Dimensionality)`.

This model is also discussed in the ngc-learn
<a href="https://ngc-learn.readthedocs.io/en/latest/museum/sparse_coding.html">documentation</a>.

## Description

This model is effectively made up of two layers -- a sensory input layer of 
error neurons that take in sensory input and compare these values against the 
sparse coding model's predictions and a latent variable layer containing the 
neurons that represent sparse codes of the data. The latent code layer connects 
to the error neurons with a synaptic cable (the "predictive synapses") that is 
adapted with a two-factor Hebbian form of plasticity. Furthermore, the error 
neurons wire to the latent codes with another synaptic cable that is restricted 
to be the transpose of the predictive synapses (i.e., they are "tied" synapses).

The dynamics that result from the above structure is a form of sensory input 
error-driven graded dynamics that are further enfored to be sparse by either a 
kurtotic prior distribution or an iterative thresholding function.

<i>Task</i>: This model engages in unsupervised form of dictionary learning and simply
learns sparse codes (and a dictionary weight matrix) that correlate with different 
input natural image pattern patches, sampled from the original images of 
(Olshausen &amp; Field, 1996).

## Hyperparameters

This model requires the following hyperparameters, tuned to produce results much akin
to that of (Olshausen &amp; Field, 1996) as well as the ISTA model:

```
T = 300 (number of discrete time steps, or E-steps, to simulate)
dt = 1 ms (integration time constant)
hid_dim = 100 (number of latent code neurons to build)
patch_shape = 16 x 16 (shape of image patches to extract)
batch_size = 250 (patch batch size to perform online E-M over)

## latent code hyper-parameters
tau_m = 20 ms (latent code time constant)

## Cauchy prior hyper-parameters
prior_type = cauchy (sparsity prior is a centered Cauchy distribution)
lmbda = 0.14 (weighting term/scale of Cauchy latent prior)

## ISTA prior hyper-parameters
threshold_type = soft_threshold (a soft thresholding function is applied to latents)
lmbda = 5e-3 (controls strength/influence of thresholding over latents)

## Hebbian plasticity/SGA hyper-parameters
eta_w = 1e-2 (global learning rate)
```

Note that a L2 norm constraint is enforced over the synaptic weight values 
inside the dictionary matrix during each M-step of the sparse coding model's 
learning process, much akin to prior work on sparse coding.
