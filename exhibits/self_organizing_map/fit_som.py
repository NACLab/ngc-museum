#import numpy as np
import sys, getopt as gopt, optparse, time
from jax import numpy as jnp, random, nn
from ngclearn.utils.viz.synapse_plot import visualize
from ngclearn.utils.io_utils import makedir

from jax import numpy as jnp, random, jit
from ngcsimlib.context import Context
import numpy as np
np.random.seed(42)

from ngclearn import Context, MethodProcess
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
## an SOM in ngc-learn is really a special kind of synaptic transform
from ngclearn.components.synapses.competitive.SOMSynapse import SOMSynapse

"""
################################################################################
Self-Organizing Map Exhibit File:

Adapts a self-organizing map (SOM) model to images pulled from the MNIST 
database. Note that an ngc-learn implementation of the SOM assumes online 
learning (one sample presented to the model at a time, iteratively). 

Usage:
$ python fit_som.py --dataX="/path/to/train_patterns.npy" --n_epochs=5 \
                    --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '', ["dataX=", "n_epochs=", "verbosity="]
)

seed = 1234
batch_size = 4 ## batch size for training; note that the ngc-learn SOM only operates online (1 pattern at a time)
n_epochs = 5 ## total number passes through dataset
dataX = "../../data/mnist/dataX.npy"
exp_dir = "exp_out/" ## experimental output directory
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--n_epochs"):
        n_epochs = int(arg.strip())
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip() 

X = jnp.load(dataX)
n_samples = X.shape[0]
print(f"> Loading {dataX}, shape: {X.shape}")
makedir(exp_dir)

## setup and run SOM
viz_mod = 2000 #5000 #2000
in_dim = X.shape[1] #15
sample_dim = int(jnp.sqrt(in_dim))

dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 3)
## set up a single layer SOM 
### NOTE: the ngc-learn SOM only operates online (1 pattern at a time)
with Context("som_ctx") as som_ctx:
    som = SOMSynapse(
        name="a",
        n_inputs=in_dim,
        n_units_x=15, ## square height = 15
        n_units_y=15, ## square width = 15
        eta=0.05, ## initial learning-rate (alpha)
        distance_function="euclidean",
        neighbor_function="ricker", 
        weight_init=dist.uniform(0., 1.),
        key=subkeys[0]
    )
    evolve_process = (MethodProcess("evolve_process")
                      >> som.evolve)
    advance_process = (MethodProcess("advance_proc")
                       >> som.advance_state)
    reset_process = (MethodProcess("reset_proc")
                     >> som.reset)

## viz initial filters / memories
W = som.weights.get()
visualize([W], [(sample_dim, sample_dim)], f"{exp_dir}som_filters_i")

## start simulation of SOM adaptation process
print(f"SOM.radius(0): {som.radius.get()[0,0]:.5f}  "
      f"SOM.eta(0): {som.eta.get()[0,0]:.5f}  "
      f"W.n(0): {jnp.linalg.norm(W)}")
for i in range(n_epochs):
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0], n_samples) ## randomly shuffle data
    # for j in range(n_samples):
    for j in range(0, n_samples - batch_size, batch_size):
        ptr = ptrs[j]
        x_n = X[ptr:ptr+batch_size, :] ## get data pattern

        ## run SOM simulation step
        reset_process.run()
        som.inputs.set(x_n) ## clamp input to SOM input layer
        advance_process.run(t=1., dt=1.)
        evolve_process.run(t=1., dt=1.)

        i_tick = som.i_tick.get() ## each sample/update counts as an "up-tick"
        print(f"\r{i}: {som.i_tick.get()} samps; "
              f"SOM.radius: {som.radius.get()[0,0]:.5f}  "
              f"SOM.eta: {som.eta.get()[0,0]:.5f}", end="")
        ## visualize synaptic efficacies of SOM
        if i_tick % viz_mod == 0:
            W = som.weights.get()
            visualize([W], [(sample_dim, sample_dim)], f"{exp_dir}som_filters_f")
    print()
    print(">> W.n = ", jnp.linalg.norm(W))
## do final visualization of synaptic efficacies of SOM
print(f"SOM.radius({som.i_tick.get()}): {som.radius.get()[0,0]:.5f}  "
      f"SOM.eta({som.i_tick.get()}): {som.eta.get()[0,0]:.5f}  "
      f"W.n({som.i_tick.get()}): {jnp.linalg.norm(W)}")
W = som.weights.get()
visualize([W], [(sample_dim, sample_dim)], f"{exp_dir}som_filters_f")

