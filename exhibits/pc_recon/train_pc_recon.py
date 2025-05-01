from jax import jit, random
import os
from ngclearn import Context, numpy as jnp
from pc_recon import PC_Recon
import sys, getopt as gopt, optparse, time


"""
################################################################################
Reconstructive Predictive Coding Exhibit File:

This mode is fit to learn latent representations of the input and reconstructs 
input data sampled from the MNIST database. 

Usage:
$ python train_rpc.py --path_data="/path/to/dataset_arrays/" 
                      --n_samples=-1 --n_iter=10


Note that there is an optional argument "--n_samples", which allows you to choose a
number less than your argument dataset's total size N for cases where you are
interested in only working with a subset of the first K samples, where K < N. 
Further note that this script assumes there is a training dataset array 
called `trainX.npy` within `path/to/dataset_arrays` as well as a testing/dev-set 
called `testX.npy`.

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 12)

# ################################################################################
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["path_data=",
                                                    "n_samples=",
                                                    "n_iter="])

path_data = "../../data/mnist/"
n_samples = -1
n_iter = 10 ## total number passes through dataset

for opt, arg in options:
    if opt in ("--path_data"):
        path_data = arg.strip()
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
print("Data Path: ", path_data)


T = 20 #30 # K = 100  ## number E-steps
dt = 1.
# batch_mod = 50
iter_mod = 1
viz_mod = 5
mb_size = 100

h3_dim, h2_dim, h1_dim = (196, 225, 256)
n_samples_test = -1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
################################################################################
## load in data and build model
jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

################################################################################
path_train = os.path.join(path_data, "trainX.npy")
x_train = jnp.load(path_train)
if n_samples > 0:
    x_train = x_train[:n_samples, :]

if n_samples > 0:
    print("-> Fitting model to only {} samples".format(n_samples))


path_test = os.path.join(path_data, "testX.npy")
x_test = jnp.load(path_test)
if n_samples > 0:
    x_test = x_test[:n_samples, :]


n_batches = int(x_train.shape[0]/mb_size)
io_dim = x_train.shape[1]
################################################################################
## initialize and compile the model with fixed hyper-parameters

model = PC_Recon(dkey, h3_dim, h2_dim, h1_dim, io_dim, batch_size=mb_size)
model.save_to_disk()
print(model.get_synapse_stats())
model.viz_receptive_fields(fname='erf_t0')
################################################################################

## begin simulation of the model using the loaded data
for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], x_train.shape[0])
    X = x_train[ptrs, :]

    n_pat_seen = 0
    print("========= Iter {}/{} ========".format(i+1, n_iter))
    L = 0
    for nb in range(n_batches):
        Xb = X[nb * mb_size: (nb + 1) * mb_size, :]        # shape: (batch_size, 784)
        Xmu, Lb = model.process(Xb, adapt_synapses=True)

        n_pat_seen = n_pat_seen + Xb.shape[0]
        L = Lb + L  ## track current global loss
        print("\r > Seen {} patterns   | n_batch {}/{}   | Train-Recon-Loss = {}".format(
            n_pat_seen, (nb+1), n_batches, L/(n_pat_seen+1)), end="")


    if i % iter_mod == 0 and i > 0:
        print()
        dkey, *subkeys2 = random.split(dkey, nb)
        X_test = x_test[random.permutation(subkeys2[0], x_test.shape[0]), :][:mb_size, :]
        Xmu_test, L_test = model.process(X_test, adapt_synapses=False) ## only perform E-steps/inference
        model.viz_recons(X_test, Xmu_test, fname=f"recons_t{(i % iter_mod)+1}")
        print("\r                                            >  Test-Recon-Loss = {} ".format(L_test / mb_size))

        model.viz_receptive_fields(fname=f"erf_t{(i % iter_mod)+1}")
        model.save_to_disk(params_only=True)  ## save final state of synapses to disk

    print()
    if (i+1) % viz_mod == 0:
        print(model.get_synapse_stats())
        model.viz_receptive_fields(fname=f"erf_t{(i % iter_mod)+1}")

## collect a test sample raster plot
model.save_to_disk(params_only=True) ## save final model parameters to disk

