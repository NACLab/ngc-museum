# Hidden Hierarchical Predictive Coding

<b>Version</b>: ngclearn==3.2.3, ngcsimlib==3.1.1

This exhibit contains an implementation of the predictive coding (PC) model specialized 
for the task of reconstruction (see the `pc_reconstruction` exhibit). This model is 
effectively a variant that embodies key characteristics across several classical 
efforts, such as:

```
Rao, Rajesh PN, and Dana H. Ballard. "Predictive coding in the visual cortex: a
functional interpretation of some extra-classical receptive-field effects." Nature neuroscience 2.1 (1999): 79-87.
```

```
Ororbia, Alexander, and Daniel Kifer. "The neural coding framework for learning 
generative models." Nature communications 13.1 (2022): 2064.
```

and, furthermore, incorporating the sparse kurtotic prior (over neural activities) from:

```
Olshausen, Bruno A., and David J. Field. "Emergence of simple-cell receptive field 
properties by learning a sparse code for natural images." Nature 381.6583 
(1996): 607-609.
```

in which the synaptic weight values are never updated and remain fixed 
at their random initialization. The model instead learns a score per
synapse, whose magnitude scoring how useful that synapse is, i.e., it 
finds a sub-network hidden within the randomly weighted PCN according 
to the edge-popup algorithm of:

```
Ramanujan, Vivek, et al. "What's hidden in a randomly weighted neural network?"
Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. 2020.
```

The base model is discussed in more details in the ngc-learn
<a href="https://github.com/NACLab/ngc-learn/blob/main/docs/museum/pc_rao_ballard1999.md">documentation</a>.


## Running the Model's Simulation

To train this implementation of PC, simply run:

```console
$ python find_hidden_pc.py --model_name="sparsePC_patch" --path_data="/path/to/dataset_arrays/" 
                           --n_samples=-1 --n_iter=10
```

Note that you can point the training script to other datasets besides the
default MNIST, just ensure that the targets inside of the directory provided 
for `path_data` are two numpy arrays of shape 
`(Number data points x D)`, one labeled `trainX.npy` (training set) and 
another labeled `testX.npy` (test/dev-set). 

## Description

This model is a four layers hierarchical predictive coding, a sensory input layer and
three internal/hidden layers of graded rate-cells (each equipped with linear 
rectifier elementwise activation functions), and one output layer for reading
out predictions of target values. Each layer connects to the next via a `ScorePatchedSynapse`, 
a synaptic cable whose weight values are initialized randomly from a signed Kaiming
constant distribution (every weight has magnitude `sqrt(2 / fan_in)` 
with a random sign) and fixed. Each synapse has a score compartment that is adaptable
via the two-factor Hebbian term (pre-synaptic term is the post-activation values
of the layer above and post-synaptic term is the error neuron post-activation values 
of the layer it predicts) multiplied by the fixed weight value and the sign of the score. 
After each adaptation on scores, the fraction `k` of synapses with the highest score 
magnitudes `|S|` in each layer forms the sub-network, while the remaining synapses are
masked out.

<i>Task</i>: This model engages in unsupervised reconstruction/representation,
learning to predict the pixel values of different input digit patterns sampled from the 
MNIST database. Note this implementation operates with patches extracted 
from the target image patterns.

## Intuition

Let `dW = pre.T@post = zF.T @ e` be the Hebbian update the weights receive 
(pre-synaptic activity times post-synaptic error), but they do not get updated.
Instead, the scores are updated as:

```
dS = sign(S) * W * dW        so that        d|S| = W * dW
```

- `W * dW > 0`: the optimizer would push `W` further in its own direction, i.e.,
  the fixed weight is already useful. `|S|` increases and the synapse has a
  higher chance to be active in the sub-network.
- `W * dW < 0`: the optimizer would push `W` towards 0. or flipits sign, so 
`|S|`decreases and the synapse has a higher chance to be pruned from the sub-network.
             
  
Per synapse, `W * dW = (zF_i W_ij) e_j`: the `W_ij` synapse's individual share of
its post node prediction, `zF_i W_ij = mu_ij` (`mu_j = sum_i mu_ij`) is compared 
with the post node's error `e_j = z_j - mu_j`. When the synaptic prediction has the
same sign as the error, i.e., it pushes `mu_j` toward `z_j`, `|S_ij|` increases 
(equivalent to `W * dW > 0`), and when the signs disagree, `|S_ij|` decreases 
(equivalent to `W * dW < 0`).

Since the weights are fixed, selecting structures becomes choosing synapses 
whose weights already point the right direction.


## Hyperparameters

This model requires the following hyperparameters, tuned to produce good-quality
receptive fields (in the bottom layer closest to the sensory input) and reconstructed 
digit input patterns:

```
T = 20 (number of time steps to simulate, or number of E-steps to take)
dt = 1 ms (integration time constant)
tau = 20 ms (rate-cell membrane time constnat)
lmbda = 0.14 (strength of Laplacian prior enforced over hidden activities; sparsePC configurations)
k = 0.5 (fraction of synapses kept in each layer's sub-network)
weight_init = signed Kaiming constant, gain sqrt(2) (fixed synaptic weight values)
## synaptic update meta-parameters
eta = 0.005 (learning rate of SGD optimizer embedded w/in each synaptic cable for the score M-step)
batch_size = 100
```