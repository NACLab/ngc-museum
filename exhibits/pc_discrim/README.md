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
$ python train_model.py --dataX="../data/baby_mnist/babyX.npy"
```

Note that you can point the training script to other datasets besides the
default MNIST, just ensure that the target for `dataX` is a numpy array of
shape `(Number data points x Pattern Dimensionality)`.

## Description

This model is effectively made up of three layers -- a sensory input layer made up
of Poisson encoding neuronal cells, a hidden layer of excitatory leaky integrators,
and another layer of inhibitory leaky integrators. The sensory layer connects to
excitatory layer with a synaptic cable that is adapted with traced-based
spike-timing-dependent plasticity. The excitatory layer connects to the inhibitory
layer with a fixed identity matrix/cable (all excitatory neurons map one-to-one to
inhibitory ones) while the inhibitory layers connects to the excitatory layer
via a fixed, negatively scaled hollow matrix/cable.

The dynamics that result from the above structure is a form of sensory input-driven
leaky integrator dynamics that are recurrently inhibited by the laterally-wired
inhibitory leaky integrators.

<i>Task</i>: This model engages in unsupervised representation learning and simply
learns sparse spike train patterns that correlate with different input digit patterns
sampled from the MNIST database.

## Hyperparameters

This model requires the following hyperparameters, tuned to produce results much akin
to that of the original Diehl and Cook model:

```
## Note: resistance scale values set to 1
tau_m_e = 100.500896468 ms (excitatory membrane time constant)
tau_m_i = 100.500896468 ms (inhibitory membrane time constant)
tau_tr= 20. ms (trace time constant)
## STDP hyper-parameters
Aplus = 1e-2 (LTD learning rate (STDP); or "nu1" in literature)
Aminus = 1e-4 (LTD learning rate (STDP); or "nu0" in literature)
w_norm = 78.4 (L1 norm constraint)
norm_T = 200 ms (time to enforce norm constraint)
factor = 22.5 (excitatory-to-inhibitory synapse scale factor)
factor = -120 (inhibitory-to-excitatory synapse scale factor)
## excitatory dynamics (with adaptive threshold)
thr = -52 mv
v_rest = -65 mv
v_reset = -60 mv
tau_theta = 1e7
theta_plus = 0.05
## inhibitory dynamics (with NO adaptive threshold)
thr = -40 mv
v_rest = -60 mv
v_reset = -45 mv
```

In effect, the model enforces a synaptic re-scaling based on an L1 norm
at the end of `200` milliseconds (ms) -- this re-scaling step is maintained
by the particular STDP synapse used in our implementation of Diehl and Cook's
model as every component have access to the simulation object's clock.

<i>Model Simplification</i>: The original Diehl and Cook model also incorporated
synaptic conductance, i.e., electrical currents were modeled by differential
equations as well, whereas our exhibit implements currents as point-wise
injections. However, if synaptic conductance is desired, one could extend the
model to use electrical currents built with ngc-learn `RateCell`s, which offer
the machinery needed for (leaky) graded/continuous valued dynamics.
