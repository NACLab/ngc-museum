# Hidden Discriminative Predictive Coding

<b>Version</b>: ngclearn==3.2.3, ngcsimlib==3.1.1

This exhibit contains an implementation of the predictive coding network (PCN)
proposed and studied in:

```
Whittington, James CR, and Rafal Bogacz. "An approximation of the error
backpropagation algorithm in a predictive coding network with local hebbian
synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.
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

## Running the Model's Simulation

To run this implementation of the hidden PCN, simply run:

```console
$ python find_hidden_pcn.py --dataX="/path/to/train_patterns.npy" \
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
label values. Each layer connects to the next via a `ScorePatchedSynapse`,
a synaptic cable whose weight values are sampled once from a signed Kaiming
constant distribution (every weight has magnitude `sqrt(2 / fan_in)` and a
random sign) and are never modified. Each synapse has a score compartment
that is adapted instead. After each adaptation, the fraction `k` of synapses
with the highest score magnitudes `|S|` in each layer forms the sub-network,
while the remaining synapses are masked out. Scores are updated via a
two-factor Hebbian rule (pre-synaptic term is the post-activation values of
the layer below and post-synaptic term is the error neuron post-activation
values of the current layer) multiplied by the fixed weight value and the
sign of the score, so a synapse enters the sub-network when its fixed weight
consistently moves the layer's input in the direction that reduces prediction
error, and leaves it otherwise. Feedback/error message passing pathways are
not learned and are set to the transpose of the corresponding forward
sub-network; the projection pathway used to initialize the latent states runs
through the same sub-network.

<i>Task</i>: This model engages in supervised/discriminative adaptation, 
learning to predict the labels of different input digit patterns sampled 
from the MNIST database.



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

This model requires the following hyperparameters:

```
T = 20 (number of time steps to simulate, or number of E-steps to take)
dt = 1 ms (integration time constant)
tau_m = 25 ms (membrane time constant of the RateCells components)
act_fx = relu (activation function used by the RateCells components)
k = 0.5 (fraction of synapses kept in each layer's sub-network)
weight_init = signed Kaiming constant, gain sqrt(2) (fixed synaptic weight values)
## synaptic update meta-parameters
eta = 0.001 (learning rate of Adam optimizer embedded w/in each synaptic cable for the score M-step)
```