# Self-Organizing Map (Kohonen, 2002)

<b>Version</b>: ngclearn>=3.0.0, ngcsimlib>=3.0.0

This exhibit contains an implementation of the self-organizing map (SOM) model 
proposed and studied in:

```
Kohonen, Teuvo. The self-organizing map. Proceedings of the IEEE 78.9: 
1464-1480 (2002).
```

## Running the Model's Simulation

To train this implementation of the SOM, simply the run convenience 
Bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for the provided MNIST  
dataset. Alternatively, you can directly run the Python training script:

```console
$ python fit_som.py --dataX="/path/to/train_patterns.npy" \
                    --n_epochs=5 --exp_dir="/path/to/sim_outputs/" \
                    --verbosity=0
```

Note that you can point the training script to other datasets besides the
default MNIST dataset (as in the Bash script); just ensure that the
target for `trainX` is a numpy array of shape 
`(Number data points x Pattern Dimensionality)`.

<!--
This model is also discussed in the ngc-learn
<a href="https://ngc-learn.readthedocs.io/en/latest/museum/self-organizing-map.html">documentation</a>.
-->

## Description

This model is effectively made up of two layers -- an input layer of 
neurons that take in sensory input which are then mapped to one of the SOM's 
internal template/centroid memories (which are arranged in a rectangular 
topological fashion). In priniple, an SOM synapse effectively performs a 
weighted form of winner-take-all competition and yields a post-activation 
vector that is the weighted neighborhood map with respect to the best-matching 
unit (BMU) for the given input pattern. This model's synaptic efficacy 
matrix is adjusted via a competitive Hebbian learning rule. 
(Note that this model does not exhibit dynamics over its activity outputs, 
there are only dynamics in terms of its internal learning properties, such 
as neighborhood radius and Hebbian learning rate that change with time / 
simulation step.) 

<i>Task</i>: This model engages in unsupervised form of clustering, yielding a 
weight matrix of "centroid" memories; these memories are related to one another 
according to a rectangular topology (thus mapped to Cartesian coordinates). An 
SOM approximately engages in a form of template matching or auto-associative 
recall. 

## Hyperparameters

This model requires the following hyperparameters:

```
## SOM topology and BMU properties
n_inputs = 784 (dimensonality of input)
n_units_x = 15 (height of rectangular topology)
n_units_y = 15 (width of rectangular topology)
distance_function = "euclidean" (distance function used to obtain BMU)
neighbor_function = "gaussian" (neighborhood function to weight around BMU)

## Hebbian plasticity/SGA hyper-parameters
eta = 0.5 (initial global learning rate, e.g., alpha)
```

Note: This implementation of the SOM assumes it adapts online/iteratively; thus, this format 
will operate only with samples presented to the model one at a time. 

