from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from dcsnn_model import DC_SNN as Model ## bring in model from museum

"""
################################################################################
Diehl and Cook Spiking Neural Network (DC-SNN) Exhibit File:

Adapts a DC-SNN to patterns from the MNIST database.

Usage:
$ python train_dcsnn.py --dataX="/path/to/train_patterns.npy" \
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
print("Data: ",dataX)

## load dataset
_X = jnp.load(dataX)
if n_samples > 0:
    _X = _X[0:n_samples,:]
    print("-> Fitting model to only {} samples".format(n_samples))
n_batches = _X.shape[0] ## num batches is = to num samples (online learning)

## basic simulation hyper-parameter/configuraiton values go here
viz_mod = 1000 #500
mb_size = 1 ## locked to batch sizes of 1
patch_shape = (28, 28)
in_dim = patch_shape[0] * patch_shape[1]

T = 200 ## num time steps to simulate (stimulus presentation window length)
dt = 1. ## integration time constant

################################################################################
print("--- Building Model ---")
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 3)
## Create model
model = Model(subkeys[1], in_dim=in_dim, T=T, dt=dt)
################################################################################
print("--- Starting Simulation ---") 

model.save_to_disk()
#sys.exit(0)

sim_start_time = time.time() ## start time profiling

print("------------------------------------")
print(model.get_synapse_stats())
## enter main adaptation loop over data patterns
for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]

    tstart = time.time()
    n_samps_seen = 0
    for j in range(n_batches):
        idx = j
        Xb = X[idx: idx + mb_size,:]

        _S = model.process(obs=Xb, adapt_synapses=True)

        n_samps_seen += Xb.shape[0]
        print("\r Seen {} images...".format(n_samps_seen), end="")
        if (j+1) % viz_mod == 0: ## save intermediate receptive fields
            tend = time.time()
            print()
            print(" -> Time = {} s".format(tend - tstart))
            tstart = tend + 0.
            print(model.get_synapse_stats())
            model.viz_receptive_fields(fname="recFields", field_shape=(28, 28))
            model.save_to_disk(params_only=True) # save final state of synapses to disk
print()

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

model.save_to_disk(params_only=True)

print("------------------------------------")
print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))

#print("****")
#model.save_to_disk() # save final state of synapses to disk
