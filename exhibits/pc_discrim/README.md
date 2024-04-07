# Discriminative Predictive Coding

<b>Version</b>: ngclearn==1.0.beta2, ngcsimlib==0.2.beta2

This exhibit contains an implementation of the predictive coding (PC) model
proposed and studied in:

Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

## Running the Model's Simulation

To train this implementation of PC, simply run:

```console
$ python train_model.py --dataX="..." --dataY="..." --devX="..." --devY="..."
```

where you replace the ellipses with paths to the appropriate numpy array 
data source. Alternatively, you may run the convenience bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for MNIST.

Note that you can point the training script to other datasets besides the
default MNIST, just ensure that the target for `dataX` is a numpy array of
shape `(Number data points x Pattern Dimensionality)`.

## Description

This model is effectively made up of four layers -- a sensory input layer,
two internal/hidden layers of graded rate-cells, and one output layer
for reading out predictions of target values, e.g., one-hot encodings of
label values. Each layer connects to the next via a simple two-factor
Hebbian synapse (pre-synaptic term is the post-activation values of
layer below and post-synaptic term is the error neuron post-activation
values of the current layer); the entire model is a simple x-to-y
hierarchical discriminative model. Feedback/error message passing pathways
are not learned and each synaptic cable's set of weight values is set to be
equal to the transpose of the corresponding forward synaptic cable's set of
weight values.

<i>Task</i>: This model engages in supervised/discriminative adaptation, learning
to predict the labels of different input digit patterns sampled from the MNIST
database.

## Hyperparameters

This model requires the following hyperparameters, tuned to produce results much akin
to that of the original Diehl and Cook model:

```
T = 10 (number of time steps to simulate, or number of E-steps to take)
dt = 0.1 ms (integration time constant)
## synaptic update meta-parameters
eta = 0.001 (learning rate of Adam optimizer embedded w/in each synaptic cable for the M-step)
w_norm = 1. (L2 norm constraint)
```

<!-- In effect, the model enforces a synaptic re-scaling based on an L2 norm
at the end of `T * dt` milliseconds (ms) -- this re-scaling step is maintained
by the particular Hebbian synapse used in our implementation of Whittington's
PC model. -->

<i>Model Simplification</i>: TODO
