from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

################################################################################
# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=",
                                                    "model_dir=", "model_type=",
                                                    "exp_dir=", "n_samples=",
                                                    "disable_adaptation=",
                                                    "param_subdir="])

model_case = "snn_case1"
disable_adaptation = True
exp_dir = "exp/"
param_subdir = "/custom"
model_type = "tistdp"
model_dir = "exp/tistdp"
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
n_samples = 10000
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ('--model_dir'):
        model_dir = arg.strip()
    elif opt in ('--model_type'):
        model_type = arg.strip()
    elif opt in ('--exp_dir'):
        exp_dir = arg.strip()
    elif opt in ('--param_subdir'):
        param_subdir = arg.strip()
    elif opt in ('--n_samples'):
        n_samples = int(arg.strip())
    elif opt in ('--disable_adaptation'):
        disable_adaptation = (arg.strip().lower() == "true")
        print(" >  Disable short-term adaptation? ", disable_adaptation)

if model_case == "snn_case1":
    print(" >> Setting up Case 1 model!")
    from snn_case1 import load_from_disk, get_nodes
elif model_case == "snn_case2":
    print(" >> Setting up Case 2 model!")
    from snn_case2 import load_from_disk, get_nodes
else:
    print("Error: No other model case studies supported! (", model_case, " invalid)")
    exit()

print(">> X: {}  Y: {}".format(dataX, dataY))

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 3)

## load dataset
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
if 0 < n_samples < _X.shape[0]:
    ptrs = random.permutation(subkeys[0], _X.shape[0])[0:n_samples]
    _X = _X[ptrs, :]
    _Y = _Y[ptrs, :]
    # _X = _X[0:n_samples, :]
    # _Y = _Y[0:n_samples, :]
    print("-> Binding {} first randomly selected samples to model".format(n_samples))
n_batches = _X.shape[0] ## num batches is = to num samples (online learning)

## basic simulation hyper-parameter/configuration values go here
viz_mod = 1000 #10000
mb_size = 1 ## locked to batch sizes of 1
patch_shape = (28, 28)
in_dim = patch_shape[0] * patch_shape[1]

T = 250 #300 ## num time steps to simulate (stimulus presentation window length)
dt = 1. ## integration time constant

################################################################################
print("--- Loading Model ---")

## Load in model
model = load_from_disk(model_dir, param_dir=param_subdir,
                       disable_adaptation=disable_adaptation)
nodes, node_map = get_nodes(model)

################################################################################
print("--- Starting Binding Process ---")

print("------------------------------------")
model.showStats(-1)

## enter main adaptation loop over data patterns
class_responses = jnp.zeros((_Y.shape[1], node_map.get("z2e").n_units))
num_bound = 0
n_total_samp_seen = 0
tstart = time.time()
n_samps_seen = 0
for j in range(n_batches):
    idx = j
    Xb = _X[idx: idx + mb_size, :]
    Yb = _Y[idx: idx + mb_size, :]

    model.reset()
    model.clamp(Xb)
    spikes1, spikes2 = model.infer(
        jnp.array([[dt * k, dt] for k in range(T)]))
    ## bind output spike train(s)
    responses = Yb.T * jnp.sum(spikes2, axis=0)
    class_responses = class_responses + responses
    num_bound += 1

    n_samps_seen += Xb.shape[0]
    n_total_samp_seen += Xb.shape[0]
    print("\r Binding {} images...".format(n_samps_seen), end="")
tend = time.time()
print()
sim_time = tend - tstart
sim_time_hr = (sim_time/3600.0) # convert time to hours
print(" -> Binding.Time = {} s".format(sim_time_hr))
print("------------------------------------")

## compute max-frequency (~firing rate) spike responses
class_responses = jnp.argmax(class_responses, axis=0, keepdims=True)
print("---- Max Class Responses ----")
print(class_responses)
print(class_responses.shape)
bind_fname = "{}binded_labels.npy".format(exp_dir)
print(" >> Saving label bindings to: ", bind_fname)
jnp.save(bind_fname, class_responses)




