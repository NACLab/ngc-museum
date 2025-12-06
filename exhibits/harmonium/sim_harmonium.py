import sys, getopt as gopt, optparse, time
from jax import numpy as jnp, random, jit
from ngclearn.utils.viz.synapse_plot import visualize
from harmonium import Harmonium

"""
################################################################################
Harmonium (Restricted Boltzmann Machine) Exhibit File:

Trains/fits a harmonium (RBM) to a dataset of sensory patterns, e.g., the MNIST
database of gray-scale images. (Assumes that a training and a dev/validation 
dataset will be provided, formatted as saved NumPy arrays.) 
Note that this simulation file treats the input data patterns as vectors of 
Bernoulli probabilities, meaning that each time data is fed into the RBM, 
binary vectors are sampled first to create discrete codes.

Usage:
$ python sim_harmonium.py --trainX="/path/to/train_patterns.npy" \
                          --devX="/path/to/validation_patterns.npy"
                          --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

def sample_binary_data(dkey, X): ## samples data to get binary codes
    dkey, *subkeys = random.split(dkey, 3)
    bX = random.bernoulli(subkeys[0], p=X, shape=X.shape)
    return dkey, bX

################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '', ["trainX=", "devX=", "verbosity="]
)

trainX_fname = "../../data/mnist/trainX.npy" ## training design matrix (dataset)
devX_fname = "../../data/mnist/validX.npy" ## development design matrix (dataset)
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--trainX"):
        trainX_fname = arg.strip()
    elif opt in ("--devX"):
        devX_fname = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("Data: ",trainX_fname)

## load in seeding dataset
X = jnp.load(trainX_fname)
#X = X * (X >= 0.45) ## binarize data
devX = jnp.load(devX_fname)
#devX = devX * (devX >= 0.45) ## binarize data

N = X.shape[0] ## number of data points
dim = X.shape[1] ## data dimensionality = P x P pixels
px = py = int(jnp.sqrt(X.shape[1])) ## assumes square input grid img dimensions

## set up general RBM simulation / training (meta-)parameters
eta = 0.0001 ## learning rate (gradient-ascent)
n_negphase_steps = 1 ## number (k) of neg-phase steps for CD-k
use_pcd = True ## should persistent CD be used? 
l1_lambda = 0. # ## L1 synaptic decay
l2_lambda = 0.01 ## L2 synaptic decay
n_iter = 100 ## epochs / number of passes through dataset
train_batch_size = 200 ## training batch-size
dev_batch_size = 5000 ## dev batch-size

## set up JAX seeding and initialize RBM model
dkey = random.PRNGKey(69)
dkey, *subkeys = random.split(dkey, 3)
model = Harmonium(
    subkeys[0], obs_dim=dim, hid_dim=256, eta=eta, l1_lambda=l1_lambda, l2_lambda=l2_lambda, 
    is_meanfield=False, use_pcd=use_pcd 
)
model.init_vis_biases(X) ## visible biases are data-dependently initialized (as per Geoff's tech-report)
model.save_to_disk()

print("--- Initial RBM Synaptic Stats ---")
print(model.get_synapse_stats())

def eval_model(X, model, batch_size, store_recon=False, verbosity=0):
    n_batches = int(X.shape[0]/batch_size) #- 1
    sptr = 0
    eptr = batch_size
    Ns = 0
    E = 0. ## total energy, E(X)
    err = 0. ## total reconstruction (squared) error, err(X)
    recon = [] ## reconstructed patterns
    for n in range(n_batches):
        x_n = X[sptr:eptr, :]
        Ns += (eptr - sptr)
        sptr += batch_size
        eptr += batch_size
        eptr = int(jnp.minimum(eptr, X.shape[0]))

        x_r_n, err_n, Ex_n, E_n = model.process(x_n, k=1, adapt_synapses=False)
        E = E_n + E
        err = err_n + err
        if store_recon:
            recon.append(x_r_n)
        if verbosity > 0:
            print(f"\r E(X) = {E/Ns} err(X) = {err/Ns} (over {Ns} samples)", end="")
    if verbosity > 0:
        print()
    if store_recon:
        recon = jnp.concat(recon, axis=0)
    return E/Ns, err/Ns, recon

## Simulate RBM fitting/training process
dkey, _devX = sample_binary_data(dkey, devX)
energy, error, xR = eval_model(_devX, model, dev_batch_size, store_recon=True)
print(f"-1| Test:  err(X) = {error:.4f}")
model.viz_receptive_fields(fname="recFields_initial", field_shape=(px, py))

visualize([xR[0:100, :].T], [(px, py)], model.exp_dir + "/filters/{}".format("recon"))

energy_im1 = energy ## energy(i-1)
n_batches = int(X.shape[0]/train_batch_size)
for i in range(n_iter): ## for every epoch
    error_i = 0.
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0], X.shape[0])
    _X = X[ptrs, :] ## shuffle data (to ensure samples i.i.d.)
    dkey, _X = sample_binary_data(dkey, _X)

    Ns = 0.
    sptr = 0
    eptr = train_batch_size
    for n in range(n_batches): ## for every batch w/in the epoch
        x_n = _X[sptr:eptr, :]
        Ns += (eptr - sptr)
        sptr += train_batch_size
        eptr += train_batch_size
        eptr = int(jnp.minimum(eptr, _X.shape[0]))

        dkey, *subkeys = random.split(dkey, 3)
        _, err_n, _, E_n = model.process(x=x_n, k=n_negphase_steps, adapt_synapses=True, dkey=subkeys[0])
        error_i = err_n + error_i
        if verbosity > 0:
            print(f"\r {i}| Train: err(X) = {error_i/Ns:.4f}  ({int(Ns)} samples)", end="")
    #model.gibbs_chain_states = None
    error_i = error_i/Ns
    if verbosity > 0:
        print()
    
    get_recon = False
    if i == (n_iter-1):
        get_recon = True

    dkey, _devX = sample_binary_data(dkey, devX)
    energy, error, xR = eval_model(_devX, model, dev_batch_size, store_recon=get_recon)
    delta_energy = jnp.abs(energy - energy_im1) ## calc abs(delta energy)
    energy_im1 = energy
    print(
        f"{i}| Dev:  |d.E(X)| = {delta_energy:.4f}  err(X) = {error:.4f}"
    )

model.save_to_disk(params_only=True) ## save final model to disk

## visualize harmonium's receptive/projective fields
model.viz_receptive_fields(fname="recFields_final", field_shape=(px, py))

## visualize some reconstruction samples
visualize([xR[0:100, :].T], [(px, py)], model.exp_dir + "/filters/{}".format("recon"))

print("--- Final RBM Synaptic Stats ---")
print(model.get_synapse_stats())

