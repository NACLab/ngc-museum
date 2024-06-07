from jax import numpy as jnp, random, nn, jit
import numpy as np, time
import sys, getopt as gopt, optparse
#from dcsnn_model import load_model
from dcsnn_model import DC_SNN as Model ## bring in model from museum
## bring in ngc-learn analysis tools
from ngclearn.utils.viz.raster import create_raster_plot


"""
################################################################################
Diehl and Cook Spiking Neural Network (DC-SNN) Exhibit File:

Visualizes the receptive fields of a trained DC-SNN model and creates a
raster plot produced by the model's layer of excitatory neurons while
processing a sampled MNIST database digit.

Usage:
$ python analyze_dcsnn.py --dataX="/path/to/train_patterns.npy" \
                          --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "sample_idx=",
                                                    "verbosity="])

sample_idx = 0 ## choose a pattern (0 <= idx < _X.shape[0])
dataX = "../../data/mnist/testX.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--sample_idx"):
        sample_idx = int(arg.strip())
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("Data: ",dataX)

## load dataset
_X = jnp.load(dataX)
n_batches = _X.shape[0]
patch_shape = (28, 28)

dkey = random.PRNGKey(time.time_ns())
model = Model(dkey=dkey, loadDir="exp/snn_stdp")
#model = load_model("exp/snn_stdp", dt=1., T=200, in_dim=_X.shape[1]) ## load in pre-trained SNN model


print("****")
## save final receptive fields
print("=> Plotting receptive fields...")
model.viz_receptive_fields(fname="recFields", field_shape=patch_shape)

print("=> Plotting raster of sample spike train...")
## create a raster plot of for sample data pattern
x_ref = _X[sample_idx:sample_idx+1,:] ## extract data pattern
_S = model.process(obs=x_ref, adapt_synapses=False, collect_spike_train=True)
cnt = jnp.sum(jnp.squeeze(_S), axis=0, keepdims=True) ## get frequencies/firing rates

neural_idx = jnp.squeeze(jnp.argmax(cnt, axis=1)) ## get highest firing rate among neurons
print(" >> Neural.Idx {} -> Input Pattern {} ".format(neural_idx, sample_idx))
field = jnp.expand_dims(model.circuit.components["W1"].weights.value[:,neural_idx], axis=1)


import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

arr = jnp.reshape(field, shape=(28,28)) #.T
plt.imshow(arr)#.T)#, interpolation='nearest')#, cmap='rgb')#, cmap='gray')
plt.savefig("exp/syn{}_digit{}.png".format(neural_idx, sample_idx))
plt.clf()

arr = jnp.reshape(x_ref, shape=(28,28)) #.T
plt.imshow(arr)#.T)#, interpolation='nearest')#, cmap='rgb')#, cmap='gray')
plt.savefig("exp/digit{}.png".format(sample_idx))
plt.clf()

create_raster_plot(_S, tag="{}".format(0),
                   plot_fname="{}/raster/z1e_raster.jpg".format(model.exp_dir))
