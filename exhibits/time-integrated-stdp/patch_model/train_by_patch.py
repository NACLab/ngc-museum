from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from ngclearn import Context
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.patch_utils import generate_patch_set

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

def save_parameters(model_dir, nodes): ## model context saving routine
    makedir(model_dir)
    for node in nodes:
        node.save(model_dir) ## call node's local save function

################################################################################
# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=", "n_samples=",
                                                    "n_iter=", "verbosity=",
                                                    "bind_target=", "exp_dir=",
                                                    "model_type=", "seed="])

seed = 1234
exp_dir = "exp/"
model_type = "tistdp"
n_iter = 1 # 10 ## total number passes through dataset
n_samples = -1
bind_target = 40000
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
    elif opt in ("--bind_target"):
        bind_target = int(arg.strip())
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()
    elif opt in ("--model_type"):
        model_type = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())
if model_type == "tistdp":
    print(" >> Setting up TI-STDP builder!")
    from tistdp_snn import build_model, get_nodes
elif model_type == "trstdp":
    print(" >> Setting up Trace-based STDP builder!")
    from trstdp_snn import build_model, get_nodes
elif model_type == "evstdp":
    print(" >> Setting up Event-Driven STDP builder!")
    from evstdp_snn import build_model, get_nodes
elif model_type == "stdp":
    print(" >> Setting up classical STDP builder!")
    from stdp_snn import build_model, get_nodes
print(">> X: {}  Y: {}".format(dataX, dataY))

## load dataset
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
if n_samples > 0:
    _X = _X[0:n_samples, :]
    _Y = _Y[0:n_samples, :]
    print("-> Fitting model to only {} samples".format(n_samples))
n_batches = _X.shape[0] ## num batches is = to num samples (online learning)

## basic simulation hyper-parameter/configuration values go here
viz_mod = 1000 #10000
mb_size = 1 ## locked to batch sizes of 1
patch_shape = (10, 10) #(28, 28)
in_dim = patch_shape[0] * patch_shape[1]
num_patches = 10

T = 250 #300 ## num time steps to simulate (stimulus presentation window length)
dt = 1. ## integration time constant

################################################################################
print("--- Building Model ---")
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 3)
## Create model
makedir(exp_dir)
model = build_model(seed, in_dim=in_dim) #Context("model") ## get model from imported header file
nodes, node_map = get_nodes(model)
model.save_to_json(exp_dir, model_type)
################################################################################
print("--- Starting Simulation ---")

sim_start_time = time.time() ## start time profiling

model.viz(name="{}{}".format(exp_dir, "recFields"), low_rez=True,
          raster_name="{}{}".format(exp_dir, "raster_plot"))

print("------------------------------------")
model.showStats(-1)

model_dir = "{}{}/custom_snapshot{}".format(exp_dir, model_type, 0)
save_parameters(model_dir, nodes)

## enter main adaptation loop over data patterns
class_responses = jnp.zeros((_Y.shape[1], node_map.get("z2e").n_units))
num_bound = 0
n_total_samp_seen = 0
for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    tstart = time.time()
    n_samps_seen = 0
    for j in range(n_batches):
        idx = j
        Xb = X[idx: idx + mb_size, :]
        Yb = Y[idx: idx + mb_size, :]

        ## generate a set of patches from current pattern
        Xb = generate_patch_set(Xb, patch_shape, num_patches, center=False)
        for p in range(Xb.shape[0]): # within a batch of patches, adapt SNN
            xs = jnp.expand_dims(Xb[p, :], axis=0)
            flag = jnp.sum(xs)
            if flag > 0.:
                model.reset()
                model.clamp(xs)
                spikes1, spikes2 = model.observe(jnp.array([[dt * k, dt] for k in range(T)]))

        n_samps_seen += Xb.shape[0]
        n_total_samp_seen += Xb.shape[0]
        print("\r Seen {} images (Binding {})...".format(n_samps_seen, num_bound), end="")
        if (j+1) % viz_mod == 0: ## save intermediate receptive fields
            tend = time.time()
            print()
            print(" -> Time = {} s".format(tend - tstart))
            tstart = tend + 0.
            model.showStats(i)
            model.viz(name="{}{}".format(exp_dir, "recFields"), low_rez=True,
                      raster_name="{}{}".format(exp_dir, "raster_plot"))

            ## save a running/current overridden copy of NPZ parameters
            model_dir = "{}{}/custom".format(exp_dir, model_type)
            save_parameters(model_dir, nodes)

    ## end of iteration/epoch
    ## save a snapshot of the NPZ parameters at this particular epoch
    model_dir = "{}{}/custom_snapshot{}".format(exp_dir, model_type, i)
    save_parameters(model_dir, nodes)
print()

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

#plot_clusters(W1, W2)

print("------------------------------------")
print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))


