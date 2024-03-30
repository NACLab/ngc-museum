# Diehl and Cook Spiking Neural Network

<b>Version</b>: ngclearn==1.0.beta0, ngcsimlib==0.2.beta1

This exhibit contains an implementation of the spiking neuronal model proposed
and studied in:

Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit recognition
using spike-timing-dependent plasticity." Frontiers in computational
neuroscience 9 (2015): 99.

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
// Note: resistance scale values set to 1
tau_m_e = 100.500896468 ms (excitatory membrane time constant)
tau_m_i = 100.500896468 ms (inhibitory membrane time constant)
tau_tr= 20. ms (trace time constant)
// STDP hyper-parameters
Aplus = 1e-2 ## LTD learning rate (STDP); nu1
Aminus = 1e-4 ## LTD learning rate (STDP); nu0
w_norm = 78.4 ## L1 norm constraint
# excitatory-to-inhibitory scale factor
factor = 22.5
# inhibitory-to-excitatory scale factor
factor = -120
# excitatory dynamics (with adaptive threshold)
thr = -52 mv
v_rest = -65 mv
v_reset = -60 mv 
tau_theta = 1e7
theta_plus = 0.05
# inhibitory dynamics (adaptive threshold disabled)
thr = -40 mv 
v_rest = -60 mv
v_reset = -45 mv
```
