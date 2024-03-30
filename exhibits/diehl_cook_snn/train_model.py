from snn_model import DC_SNN_Model as Model
from jax import numpy as jnp, random
import sys, getopt, optparse

## bring in ngc-learn analysis tools
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.patch_utils import generate_patch_set


# read in configuration file and extract necessary simulation variables/constants
options, remainder = getopt.getopt(sys.argv[1:], '', ["dataX="])
# GPU arguments
dataX = "../data/baby_mnist/babyX.npy"
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()

## load dataset
_X = jnp.load(dataX)
n_batches = _X.shape[0]

viz_mod = 1
n_samp_mod = 100 #50
mb_size = 1
n_iter = 10 #1
patch_shape = (28, 28)
in_dim = patch_shape[0] * patch_shape[1]

T = 200 #250 # num discrete time steps to simulate
dt = 1.

dkey = random.PRNGKey(1234)
#dkey, *subkeys = random.split(dkey, 10)

################################################################################
## Create model
model = Model(dkey, in_dim=in_dim, T=T, dt=dt)
################################################################################

print("****")
model.viz_receptive_fields(fname="start", field_shape=patch_shape)

x_ref = _X[0:1,:]
_S = model.process(obs=x_ref, adapt_synapses=False, collect_spike_train=True)
create_raster_plot(jnp.concatenate(_S,axis=0).T, tag="{}".format(0),
                   plot_fname="{}/raster/z1e_raster_i.jpg".format(model.exp_dir))

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]

    n_samps_seen = 0
    for j in range(n_batches):
        idx = j #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]

        _S = model.process(obs=Xb, adapt_synapses=True)

        n_samps_seen += Xb.shape[0]
        print("\r Seen {} images...".format(n_samps_seen), end="")
        if n_samps_seen % n_samp_mod == 0:
            print()
            model.viz_receptive_fields(fname="tmp", field_shape=patch_shape)
    if (i+1) % viz_mod == 0:
        print()
        model.viz_receptive_fields(fname=str(i+1), field_shape=patch_shape)
print()
print("****")

model.viz_receptive_fields(fname="final", field_shape=patch_shape)
model.save_to_disk() # save final state of synapses to disk

x_ref = _X[0:1,:]
_S = model.process(obs=x_ref, adapt_synapses=False, collect_spike_train=True)
create_raster_plot(jnp.concatenate(_S,axis=0).T, tag="{}".format(0),
                   plot_fname="{}/raster/z1e_raster_f.jpg".format(model.exp_dir))
