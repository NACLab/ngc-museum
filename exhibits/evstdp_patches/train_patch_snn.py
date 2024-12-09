from jax import numpy as jnp, random
import sys, getopt as gopt
## bring in ngc-learn tools
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.patch_utils import generate_patch_set
from snn import SNN as Model ## bring in model from museum

"""
################################################################################
Event-STDP trained Spiking Neural Network Exhibit File:

Adapts an SNN to 10x10 patches extracted from the MNIST database.

Usage:
$ python train_patch_snn.py --dataX="/path/to/train_patterns.npy" \
                            --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "n_samples=",
                                                    "n_iter=", "verbosity="])

n_iter = 1 # 10 ## total number passes through dataset
n_samples = -1
dataX = "../../data/mnist/trainX.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
print("Data: ", dataX)

viz_mod = 1
batch_mod = 50 #100 #50
num_patches = 10
mb_size = 1
n_iter = 1 #10
patch_shape = (10, 10)
in_dim = patch_shape[0] * patch_shape[1]
hid_dim = 100 #64
T = 50 #100
dt = 3.

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

_X = jnp.load(dataX)
if n_samples > 0:
    _X = _X[0:n_samples,:]
    print("-> Fitting model to only {} samples".format(n_samples))
n_batches = _X.shape[0]

model = Model(subkeys[0], in_dim=in_dim, hid_dim=hid_dim, T=T, dt=dt)
model.save_to_disk()

print(model.get_synapse_stats())

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]

    n_pat_seen = 0
    print("========= Iter {} ========".format(i))
    for j in range(n_batches):
        idx = j #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]
        ## generate a set of patches from current pattern
        Xb = generate_patch_set(Xb, patch_shape, num_patches, center=False)

        for p in range(Xb.shape[0]): # within a batch of patches, adapt SNN
            xs = jnp.expand_dims(Xb[p,:], axis=0)
            flag = jnp.sum(xs)
            if flag > 0.:
                _S = model.process(obs=xs, adapt_synapses=True)
                n_pat_seen += 1

        print("\r > Seen {} patches so far ({} patterns)".format(n_pat_seen, (j+1)),end="")
        if j % batch_mod == 0:
            print()
            model.viz_receptive_fields(fname="recFields", field_shape=patch_shape)
            model.save_to_disk(params_only=True) # save final state of synapses to disk
            print(model.get_synapse_stats())
    print()
    if (i+1) % viz_mod == 0:
        model.viz_receptive_fields(fname="recFields", field_shape=patch_shape)

## collect a test sample raster plot
model.save_to_disk(params_only=True)

Xb = generate_patch_set(_X[None, 0, :], patch_shape, num_patches, center=False)
for p in range(Xb.shape[0]):
    xs = jnp.expand_dims(Xb[p,:], axis=0)
    _S = model.process(obs=xs, adapt_synapses=True)
    flag = jnp.sum(_S)
    if flag > 0.:
        print("# spikes emitted = ",flag)
        create_raster_plot(_S, tag="", plot_fname="{}/raster/z1.jpg".format(model.exp_dir), title_font_size=30)
        break
