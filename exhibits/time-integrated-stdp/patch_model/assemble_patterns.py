from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from ngclearn.utils.io_utils import makedir
from custom.patch_utils import Create_Patches
from ngclearn.utils.viz.synapse_plot import visualize

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

################################################################################
# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=", "n_samples=",
                                                    "verbosity=", "exp_dir=", "model_dir=",
                                                    "model_type=", "param_subdir=", "seed="])

seed = 1234
exp_dir = "exp/"
model_type = "tistdp"
model_dir = ""
param_subdir = "/custom"
n_samples = -1
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
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()
    elif opt in ('--param_subdir'):
        param_subdir = arg.strip()
    elif opt in ("--model_type"):
        model_type = arg.strip()
    elif opt in ("--model_dir"):
        model_dir = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())

if model_type == "tistdp":
    print(" >> Setting up TI-STDP Patch-Model builder!")
    from patch_tistdp_snn import load_from_disk, get_nodes
else:
    print(" >> Model type ", model_type, " not supported!")

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
viz_mod = 100 # 1000 #10000
mb_size = 1 ## locked to batch sizes of 1
patch_shape = (28, 28) ## same as image_shape
py = px = 28
img_shape = (py, px)
in_dim = patch_shape[0] * patch_shape[1]

x = jnp.ones(patch_shape)
pty = ptx = 7
in_patchShape = (pty, ptx) #(14, 14) #(7, 7)
in_patchSize = pty * ptx
stride_shape = (2, 2) #(0, 0)
patcher = Create_Patches(x, in_patchShape, stride_shape)
x_pat = patcher.create_patches(add_frame=False, center=False)
n_in_patches = patcher.nw_patches * patcher.nh_patches

z1_patchSize = 4 * 4
z1_patchCnt = 16

T = 250 #300 ## num time steps to simulate (stimulus presentation window length)
dt = 1. ## integration time constant

################################################################################
print("--- Building Model ---")
## Create model
makedir(exp_dir)
## get model from imported header file
disable_adaptation = False # True
model = load_from_disk(model_dir, param_dir=param_subdir,
                       disable_adaptation=disable_adaptation)
nodes, node_map = get_nodes(model)
################################################################################
print("--- Starting Simulation ---")

sim_start_time = time.time() ## start time profiling

print("------------------------------------")
model.showStats(-1)

## sampling concept
K = 300 #100 #400 #200 #100
W2 = node_map.get("W2").weights.value
W1 = node_map.get("W1").weights.value

print(">> Visualizing top-level filters!")
n_neurons = int(jnp.sqrt(W2.shape[0]))
visualize([W2], [(n_neurons, n_neurons)], "{}_toplevel_filters".format(exp_dir))

print(" >> Building Level 1 Block Filter Tensor...")
W1_images = []
for i_ in range(W1.shape[1]):
    img_i = [] ## combine along row axis (in cache)
    strip_i = [] ## combine along column axis
    for j_ in range(z1_patchCnt):
        _filter = W1[in_patchSize * j_:(j_ + 1) * in_patchSize, i_:i_ + 1]
        _filter = jnp.reshape(_filter, (ptx, pty)) # reshape to patch grid
        strip_i.append(_filter)
        if len(strip_i) >= int(py/pty):
            img_i.append(jnp.concatenate(strip_i, axis=1))
            strip_i = []
    img_i = jnp.concatenate(img_i, axis=0)
    # plt.imshow(img_i)
    # plt.savefig("exp_evstdp_1234/test/filter{}.jpg".format(i_))
    print("\r {} filters built...".format(i_),end="")
    W1_images.append(img_i)
    img_i = [] ## clear cache
print()

print(" >> Building super-imposed sampled images!")
samples = []
for i in range(W2.shape[1]):
    W2_i = W2[:, i]
    indices = jnp.argsort(W2_i, descending=True)[0:K]
    coefficients = W2_i[indices]
    #indices = jnp.where(W2_i > thr)
    #Z = jnp.amax(W2_i)
    #indices = jnp.flip(jnp.argsort(W2_i))[0:K]
    #coefficients = W2_i[indices]#/Z

    xSample = 0.
    ptr = 0
    for idx in indices: # j in range(indices.shape[0]):
        coeff_i = coefficients[ptr]
        Ki = W1_images[idx] * coeff_i
        xSample += Ki
        ptr += 1
    xSample = xSample.reshape(1, -1).T
    samples.append(xSample)
    print("\r Crafted {} samples...".format(len(samples)), end="")
print()
samples = jnp.concatenate(samples, axis=1)

visualize([samples], [(28, 28)], "{}{}".format(exp_dir, "samples"))


