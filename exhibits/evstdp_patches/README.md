# Event-Based STDP Spiking Neural Network

<b>Version</b>: ngclearn==1.1.beta1, ngcsimlib==0.3.beta1

This exhibit contains an implementation of the spiking neuronal model proposed
and studied in:

Tavanaei, Amirhossein, Timothée Masquelier, and Anthony Maida.
"Representation learning using event-based STDP." Neural Networks 105
(2018): 294-303.

## Running the Model's Simulation

To train this implementation of Tavanaei's model, simply run:

```console
$ python train_patch_snn.py --dataX="../data/mnist/trainX.npy" --n_samples=10000
```

Note that you can point the training script to other datasets besides the
default MNIST, just ensure that the target for `dataX` is a numpy array of
shape `(Number data points x Pattern Dimensionality)`.

Alternatively, you may run the convenience bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for MNIST.

<!--
<p align="center">
  <img height="350" src="fig/evstdp_arch.jpg"><br>
  <i>Visual depiction of the DC-SNN architecture.</i>
</p>

This model is also discussed in the ngc-learn
<a href="https://ngc-learn.readthedocs.io/en/latest/museum/snn_patches.html">documentation</a>.
-->

## Description

This model is effectively made up of three layers -- a sensory input layer made up
of Bernoulli encoding neuronal cells, an exponential (spike-response) kernel that processes 
the Bernoulli cells to produce EPSP pulses, and a hidden layer of `64` winner-take-all 
spiking (WTAS) cells. The sensory layer outputs (from the kernel) 
connect to the WTAS layer with a synaptic cable that is adapted with event-based
spike-timing-dependent plasticity (evSTDP). 

The dynamics that result from the above structure is a form of sensory input-driven
winner-take-all dynamics that are exhibit lateral inhibition through 
a softmax function inside the WTAS voltage value calculation.

<i>Task</i>: This model engages in unsupervised representation learning and simply
learns sparse spike train patterns that correlate with different input `10 x 10` 
patches sampled from digit image images in the MNIST database.

## Hyperparameters

This model requires the following hyperparameters:

```
T = 50 (number of discrete time steps to simulate)
dt = 3 ms (integration time constant)
## WTAS hyper-parameters
### Note: resistance scale values set to 1
tau_m = 100 ms (WTAS membrane time constant)
refract_T = 5 ms
thrBase = 0.2 (base voltage threshold value)
thr_gain = 0.002 (increment to apply to adaptive thresholds)
refract_T = 5 (relative refractory period)
thr_jitter = 0.05 (initialization noise applied to threshold inits)
## STDP hyper-parameters
eta_w = 0.0055 (evSTDP global learning rate)
lmbda = 0.01
w_bound = 1.
## spike-response kernel dynamics
tau_w = 0.5 
nu = 4.
```

<i>Model Extension</i>: This implemented model also features a simple 
modification to the original model by adding in a relative refractory 
period mechanism to the WTAS cell. In this simulation, we enforce 
further sparsity in the WTAS layer by setting the refractory period 
to be five milliseconds.
