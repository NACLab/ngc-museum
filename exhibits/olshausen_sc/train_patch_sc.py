from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time

## bring in ngc-learn tools
from ngclearn.utils.patch_utils import generate_patch_set
from sparse_coding import SparseCoding as Model ## bring in model from museum

"""
################################################################################
Sparse Coding Exhibit File:

Adapts a sparse coding linear generative model to 10x10 patches extracted from 
the MNIST database. Note that depending on the input arguments to this script, 
one can either train a sparse coding model with a Cauchy prior over latents or 
an ISTA model.

Usage:
$ python train_patch_sc.py --dataX="/path/to/train_patterns.npy" --n_iter=200 \
                           --model_type="<model_choice>" --verbosity=0
                           
Note that there is an optional argument "--n_samples", which allows to choose a 
number less than your argument dataset's total size N for cases where you are 
interested in only working with a subset of the first K samples, where K < N.

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

################################################################################
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "n_samples=",
                                                    "n_iter=", "model_type=", 
                                                    "verbosity="])

model_type = "sc_cauchy" # ista
n_iter = 200 # 10 ## total number passes through dataset
n_samples = -1
dataX = "../../data/natural_scenes/dataX.npy"
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
    elif opt in ("--model_type"):
        model_type = arg.strip()
print("Data: ", dataX)

viz_mod = 1
batch_mod = 50
num_patches = 250 ## number of patches in a batch
mb_size = 1 ## number of images in a batch (fixed to 1)
patch_shape = (16, 16)
in_dim = patch_shape[0] * patch_shape[1]
hid_dim = 100
T = 300
dt = 1.

################################################################################
## load in data and build model
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

_X = jnp.load(dataX)
if n_samples > 0:
    _X = _X[0:n_samples,:]
    print("-> Fitting model to only {} samples".format(n_samples))
n_batches = _X.shape[0]

model = Model(subkeys[0], in_dim=in_dim, hid_dim=hid_dim, T=T, dt=dt,
              batch_size=num_patches, model_type=model_type)
model.save_to_disk()

print(model.get_synapse_stats())
model.viz_receptive_fields(fname="recFields_init", field_shape=patch_shape)

################################################################################
## begin simulation of the model using the loaded data

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]

    n_pat_seen = 0
    print("========= Iter {} ========".format(i))
    L = 0.
    for j in range(n_batches):
        idx = j
        Xb = X[idx: idx + mb_size, :]
        ## generate a set of patches from current pattern
        Xb = generate_patch_set(Xb, patch_shape, num_patches, center=True)

        xs_mu, Lb = model.process(obs=Xb, adapt_synapses=True)
        n_pat_seen += Xb.shape[0]

        L = Lb + L ## track current global loss
        print("\r > Seen {} patches so far ({} patterns); L = {}".format(n_pat_seen,
                                                                        (j+1), (L/(j+1) * 1.)), end="")
        if j % batch_mod == 0 and j > 0:
            print()
            model.viz_receptive_fields(fname="recFields", field_shape=patch_shape)
            model.save_to_disk(params_only=True) # save final state of synapses to disk
            print(model.get_synapse_stats())
    print()
    if (i+1) % viz_mod == 0:
        model.viz_receptive_fields(fname="recFields", field_shape=patch_shape)

## collect a test sample raster plot
model.save_to_disk(params_only=True)
