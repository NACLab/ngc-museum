# Harmonium (Restricted Boltzmann Machine)

<b>Version</b>: ngclearn==3.0.0, ngcsimlib==3.0.0

This exhibit contains an implementation of a harmonium, or restricted Boltzmann machine (RBM), trained via contrastive divergence for reconstruction and generative modeling tasks. This model features the core model presented in: 

```
Hinton, Geoffrey E. "Training products of experts by maximizing contrastive likelihood." Technical Report, Gatsby computational neuroscience unit (1999).

Rumelhart, David E., James L. McClelland, and PDP Research Group. Parallel distributed processing, volume 1: Explorations in the microstructure of cognition: Foundations. The MIT press, 1986. (Chapter 6: information processing in dynamical systems: foundations of harmony theory parallel distributed processing, Paul Smolensky)
```

built in accordance with many of the pragmatic recommendations found within: 

```
Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." Neural Networks: Tricks of the Trade: Second Edition. Berlin, Heidelberg: Springer Berlin Heidelberg, 2012. 599-619.
```

<p align="center">
  <img height="200" src="fig/harmonium.jpg"><br>
  <i>Visual depiction of the harmonium architecture, which contains one hidden layer $\mathbf{h}$ (unobserved/latent variables) and visible layer $\mathbf{v}$ (observed variables).</i>
</p>

This model is also discussed in the ngc-learn <a href="https://ngc-learn.readthedocs.io/en/latest/museum/harmonium.html">documentation</a>.

## Running the Model's Simulation

To train this implementation of an RBM, simply run:

```console
$ python sim_harmonium.py --seed=1234 --results_dir=/output_dir/ 
```

Make sure you create the relevant results directory you provide as the argument to `results_dir`.

Alternatively, you may run the convenience bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for MNIST. Furthermore, the above script will execute a process for sampling the RBM, which can also be done separately via:

```console
$ python sample_harmonium.py
```

Note that you can point the training script to other datasets besides the default MNIST; just ensure that the targets for `dataX` and `devX` are numpy arrays of shape `(Number data points x D)` for data patterns (i.e., `dataX` and `devX`).

## Description

This model is effectively made up of two layers of stochastic binary (Bernoulli-distributed) neurons -- a sensory input layer (observed variables) and a single hidden neurons (unobserved variables). The sensory layer connects to the hidden one via a set of synapses that adapt their efficacies according to contrastive divergence (or contrastive Hebbian learning). Note that the hidden layer technically (recurrently) connects back to the input layer via another set of synapses; however, these specific synapses are "tied" to the ones that wire from the input to hidden layers (i.e., the hidden-to-input synaptic weight matrix is equal to the transpose of the input-to-hidden synaptic weight matrix). Both input and hidden layers further feature bias parameters (i.e., this means there are learnable visible and hidden bias values).

<i>Task</i>: This model engages in unsupervised reconstruction, learning to predict the pixel values of different input digit patterns sampled from the MNIST database. It is further used as a generative model; samples may be drawn from it via a constructed block Gibbs sampler. 

## Hyperparameters

This model, set to use `256` hidden binary neurons, requires the following hyperparameters, tuned to produce good-quality receptive fields, reconstruct digit input patterns, and produce reasonable-quality samples through block Gibbs sampling:

```
## synaptic update meta-parameters
eta = 0.0001 (learning rate of Adam optimizer that applies CD updates to synapses)
n_negphase_steps = 1 (number of contrastive divergence -- CD -- negative-phase steps, k, to take)
use_pcd = True (forces this harmonium to use a form of persistent CD learning)
l1_lambda = 0. (strength of Laplacian prior enforced over synaptic weights)
l2_lambda = 0.01 (strength of Gaussian prior enforced over synaptic weights) 
n_iter = 100 (number of epochs/iterations)
train_batch_size = 500 (number of samples to use w/in a CD-batch update)
```

Note that, for the block Gibbs sampler, we utilized the following setting to produce reasonable confabulations from the RBM:

```
thinning_point = 20 (how many Gibbs sampling steps must be taken before storing a model sample)
n_samps = 9 * 9 (how many total model samples are to be collected)
burn_in = 10 * thinning_point (number of burn-in steps to take in a new MCMC chain)
```

where we note that "thinning" was used to obtain samples at regular intervals within the Gibbs sampling chain and burn-in was employed to reach a reasonable point within the chain (equilibrium) to obtain correct samples from the RBM's underlying distribution. Each Markov chain within the Gibbs sampler is initialized from random binary codes, unless the persistent chains are used; in the last case, sampling begins at the current state of the persistent Gibbs chains. 

