# Discriminative Predictive Coding

<b>Version</b>: ngclearn==1.1.beta1, ngcsimlib==0.3.beta1

This exhibit contains an implementation of the predictive coding (PC) model (
also known as the predictive coding network or PCN) proposed and studied in:

Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

<p align="center">
  <img height="350" src="fig/pcn_arch.jpg"><br>
  <i>Visual depiction of the PCN architecture.</i>
</p>

This model is also discussed in the ngc-learn
<a href="https://ngc-learn.readthedocs.io/en/latest/museum/pcn_discrim.html">documentation</a>.

## Running the Model's Simulation

To train this implementation of PC, simply run:

```console
$ python train_pcn.py --dataX="/path/to/train_patterns.npy" \
                      --dataY="/path/to/train_labels.npy" \
                      --devX="/path/to/dev_patterns.npy" \
                      --devY="/path/to/dev_labels.npy" \
                      --verbosity=0
```


Alternatively, you may run the convenience bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for MNIST.

Note that you can point the training script to other datasets besides the
default MNIST, just ensure that the targets for `dataX`, `dataY`, `devX`, and
`devY` are numpy arrays of shape `(Number data points x D)` for data patterns  
(i.e., `dataX` and `devX`) and shape `(Number data points x C)` for labels
(`dataY` and `devY`).

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
to that of the original Whittington et al. model:

```
T = 10 (number of time steps to simulate, or number of E-steps to take)
dt = 0.1 ms (integration time constant)
## synaptic update meta-parameters
eta = 0.001 (learning rate of Adam optimizer embedded w/in each synaptic cable for the M-step)
```
