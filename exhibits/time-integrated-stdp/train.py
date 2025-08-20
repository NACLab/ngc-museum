from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
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
options, remainder = gopt.getopt(
    sys.argv[1:], '', ["dataX=", "dataY=", "n_samples=", "n_iter=", "verbosity=",
                       "bind_target=", "exp_dir=", "model_type=", "seed=",
                       "use_patches=", "model_case="]
)

model_case = "snn_case1"
seed = 1234
exp_dir = "exp/"
model_type = "tistdp"
n_iter = 1 # 10 ## total number passes through dataset
n_samples = -1
bind_target = 40000
use_patches = False
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
    elif opt in ("--model_case"):
        model_case = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--use_patches"):
        use_patches = int(arg.strip()) #(arg.strip().lower() == "true")
        use_patches = (use_patches == 1)

if model_case == "snn_case1":
    print(" >> Setting up Case 1 model!")
    from snn_case1 import build_model, get_nodes
elif model_case == "snn_case2":
    print(" >> Setting up Case 2 model!")
    from snn_case2 import build_model, get_nodes
else:
    print("Error: No other model case studies supported! (", model_case, " invalid)")
    exit()

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
patch_shape = (28, 28) ## same as image_shape
num_patches = 10
if use_patches:
    patch_shape = (10, 10)
in_dim = patch_shape[0] * patch_shape[1]

T = 250 #300 ## num time steps to simulate (stimulus presentation window length)
dt = 1. ## integration time constant

################################################################################
print("--- Building Model ---")
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 3)
## Create model
makedir(exp_dir)
## get model from imported header file
model = build_model(seed, in_dim=in_dim, is_patch_model=use_patches, algo=model_type)
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

        if use_patches:
            ## generate a set of patches from current pattern
            X_patches = generate_patch_set(Xb, patch_shape, num_patches, center=False)
            for p in range(X_patches.shape[0]):  # within a batch of patches, adapt SNN
                xs = jnp.expand_dims(X_patches[p, :], axis=0)
                flag = jnp.sum(xs)
                if flag > 0.:
                    model.reset()
                    model.clamp(xs)
                    spikes1, spikes2 = model.observe(
                        jnp.array([[dt * k, dt] for k in range(T)]))

                    if n_total_samp_seen >= bind_target:
                        responses = Yb.T * jnp.sum(spikes2, axis=0)
                        class_responses = class_responses + responses
                        num_bound += 1
        else:
            model.reset()
            model.clamp(Xb)
            spikes1, spikes2 = model.observe(jnp.array([[dt * k, dt] for k in range(T)]))
            # print(tr1)
            # print("...")
            # print(jnp.sum(spikes1, axis=0))
            # print(jnp.sum(spikes2, axis=0))
            # exit()

            if n_total_samp_seen >= bind_target:
                responses = Yb.T * jnp.sum(spikes2, axis=0)
                class_responses = class_responses + responses
                num_bound += 1

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

# print(" >> Producing final rec-fields / raster plots!")
# model.viz(name="{}{}".format(exp_dir, "recFields"), low_rez=False,
#                       raster_name="{}{}".format(exp_dir, "raster_plot"))

class_responses = jnp.argmax(class_responses, axis=0, keepdims=True)
print("---- Max Class Responses ----")
print(class_responses)
print(class_responses.shape)
jnp.save("{}binded_labels.npy".format(exp_dir), class_responses)

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

print("------------------------------------")
print(" Trial.sim_time = {} h  ({} sec)".format(sim_time_hr, sim_time))


