from jax import jit, random
import os
from ngclearn import Context, numpy as jnp
from hierarchical_pc import HierarchicalPredictiveCoding
import sys, getopt as gopt, optparse, time
from ngclearn.components.input_encoders.ganglionCell import create_patches

import sympy


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



# ═══════════════════════════════════════════════════════════════════════════
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["path_data=",
                                                    "n_samples=",
                                                    "n_iter="])

experiment_circuit_name = "pc_mlp"
dataset_name = "/mnist"
path_data = "../../data/" + dataset_name
exp_dir = "exp/" + experiment_circuit_name + dataset_name

n_samples = -1
n_iter = 10                         ## total number passes through dataset
iter_mod = 1

for opt, arg in options:
    if opt in ("--path_data"):
        path_data = arg.strip()
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
print("Data Path: ", path_data)

# ═══════════════════════════════════════════════════════════════════════════
jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, n_iter)
# ═══════════════════════════════════════════════════════════════════════════
# Training Configuration
shuffle = True
mb_size = 100
mb_vis_size = 100

# ═══════════════════════════════════════════════════════════════════════════
## load the data
img_train = jnp.load(os.path.join(path_data, "trainX.npy"))
y_train = jnp.load(os.path.join(path_data, "trainY.npy"))

img_test = jnp.load(os.path.join(path_data, "testX.npy"))
y_test = jnp.load(os.path.join(path_data, "testY.npy"))

# sequential classes (not shuffled)
img_train = img_train[jnp.argsort(jnp.argmax(y_train, axis=1))]
img_test = img_test[jnp.argsort(jnp.argmax(y_test, axis=1))]

image_size = img_train.shape[1]
ix = iy = int(jnp.sqrt(image_size))
image_shape = (ix, iy)

img_train = img_train.reshape(-1, *image_shape)
img_test = img_test.reshape(-1, *image_shape)

# ════  Stimuli Configuration  ══════════════════════════════════════════════
area_shape = image_shape                    ## (ax, ay) = (ix, iy): full image
(px, py) = patch_shape = image_shape        ## (ix, iy): full image
(sx, sy) = step_shape = patch_shape         ## (sx, sy) --- ix = px + (nx-1) * sx

n_cells = 1                     ## ==1 means full image at the time image
n_p1 = 1                        ## number of h1 patches/PE-modules
n_p2 = 1                        ## number of h2 patches/PE-modules
n_p3 = 1                        ## number of h3 patches/PE-modules

p3_size = 32                    ## h3 patch dimension
p2_size = 128                   ## h2 patch dimension
p1_size = 64                    ## h1 patch dimension
pin_size = patch_shape[0] * patch_shape[1]     ## input patch dim (== h1 neurons receptive field size)

## ═══════════════════════════════════════════════════════════════════════════
## Computed Dimensions
h3_dim = p3_size * n_p3
h2_dim = p2_size * n_p2                        # = 128 × 1  = 128
h1_dim = p1_size * n_p1                        # =  32 × 3  = 96
in_dim = pin_size * n_cells                   # = 256 × 3  = 768

# print("tot_patch_per_image is ", tot_patch_per_image)
# # print("number of input patches is ", n_inPatch)
# if tot_patch_per_image / n_inPatch != int(tot_patch_per_image / n_inPatch):
#     print("recommended mb_size and n_inPatch")
#     input(sympy.factorint(tot_patch_per_image, multiple=True))


# if patch_shape == image_shape:    ## Full image case
#     images_per_batch = mb_size
# else:                             ## Patch case
#     n_samples = 1000              ## reduce computational cost
#     mb_size = tot_patch_per_image // n_inPatch
#     images_per_batch = 1

## ══════════════════════════════════════════════════════════════════════════
## Energy Dynamics
T = 30                                      ## number E-steps
dt = 1.

## ══════════════════════════════════════════════════════════════════════════
## split the full image into local views for retinal ganglion cells local receptive fields
x_train = create_patches(img_train, patch_shape=area_shape, step_shape=area_shape) ### shape: (N | n_areas | (area_shape))
x_test = create_patches(img_test, patch_shape=area_shape, step_shape=area_shape)   ### shape: (N | n_areas | (area_shape))

x_train = x_train.reshape(-1, *area_shape)                                      ### shape: (n_total_obs | (area_shape))
x_test = x_test.reshape(-1, *area_shape)                                        ### shape: (n_total_obs | (area_shape))

# ═══════════════════════════════════════════════════════════════════════════
################################################################################
## initialize and compile the model with fixed hyper-parameters
model = HierarchicalPredictiveCoding(dkey,
                                     circuit_name=experiment_circuit_name,
                                     h3_dim=h3_dim, h2_dim=h2_dim, h1_dim=h1_dim, in_dim=in_dim,
                                     n_p3=n_p3, n_p2=n_p2, n_p1=n_p1, n_inPatch=n_cells,
                                     area_shape = area_shape,
                                     patch_shape = patch_shape,
                                     step_shape = step_shape,
                                     batch_size = mb_size,
                                     T=T, dt=dt,
                                     tau_m=20,
                                     lr=0.005,
                                     act_fx = "relu",
                                     r3_prior = ("laplacian", 0.14),
                                     r2_prior = ("laplacian", 0.14),
                                     r1_prior = ("laplacian", 0.14),
                                     synaptic_prior=("ridge", 0.02),
                                     exp_dir=exp_dir, reset_exp_dir=True
                                     )

model.save_to_disk()          # NOTE: save initial model parameters to disk, uncomment this line if we are loading a saved model
model.load_from_disk(exp_dir) # NOTE: uncomment this line and comment the above lines to load a saved model
print(model.get_synapse_stats())
model.viz_receptive_fields(max_n_vis=mb_vis_size, fname='erf_t0')

# ═══════════════════════════════════════════════════════════════════════════
## begin simulation of the model using the loaded data
if n_samples > 0:
    x_train = x_train[:n_samples, :]
    print("-> Fitting model to only {} samples".format(n_samples))

n_batch_train = x_train.shape[0] // mb_size
n_batch_test = x_test.shape[0] // mb_size
ptrs_ = random.permutation(subkeys[1], x_test.shape[0])
X_test = x_test[ptrs_, :]

for i in range(n_iter):
    # ════════════════════  shuffle   ═════════════════════
    if shuffle:
        ptrs = random.permutation(subkeys[0], x_train.shape[0])
        X = x_train[ptrs, :]
    else:
        X = x_train

    # ═══════════════════════════════════════════════════════════════════════════
    n_seen = 0
    cumultive_loss = 0
    for nb in range(n_batch_train):
        # ════════════════════  get data batch  ═════════════════════
        mb = nb * mb_size
        Xb = X[mb:mb + mb_size, :]       # Extract batch: (B | ax, ay)

        # ═══════════════════ Model Processing ══════════════════════
        Xmu = model.process(Xb, adapt_synapses=True)

        # ════════════════════   Metric Update  ═════════════════════
        Lb = model.e0.L.get()                ## batch reconstruction loss
        n_seen += Xb.shape[0]                ## Total patterns seen
        cumultive_loss += Lb                 ## Accumulate loss
        avg_loss = cumultive_loss / (n_seen + 1)

        # ═══════════════════   Progress Display  ════════════════════
        print( f"\r "
            f"│ Iter: {i:>1} "
            f"│ Seen: {n_seen:>6} patterns "
            f"│ Batch: {nb + 1:>4}/{n_batch_train:<4} "
            f"│   Train-Loss: {avg_loss:>7.4f} ",
            end="", flush=True
        )

    # ═══════════════════════════════════════════════════════════════════════════
    if (i+1) % iter_mod == 0:
        # ═══════════════════   L1 Synaptic Filters Display  ════════════════════
        model.viz_receptive_fields(max_n_vis=mb_vis_size, fname=f"erf_t{i+1}")


        ## ════════════════════════════════════════════════════════════════════
        ##################   test phase
        X_vis_test = X_test[:mb_vis_size, :]

        test_loss = 0
        for nb_t in range(n_batch_test):
            # ════════════════════  get data batch  ═════════════════════
            mb_t = nb_t * mb_size
            xb_test = X_test[mb_t:mb_t + mb_size, :]  # Extract batch: (B | ax, ay)

            # ═══════════════════ Infer Test Data ══════════════════════
            Xbt_mu = model.process(xb_test, adapt_synapses=False)

            # ════════════════════  Test Metric  ═════════════════════
            test_loss += model.e0.L.get()

        ## ═══════════════════  Test Metrics Display  ════════════════════
        print(f"│ Test-Loss: {test_loss / len(X_test):>7.4f}  ",
              f"│\n")
        ## ═══════════════════ Show Test Reconstruction  ════════════════════
        model.viz_recons(X_test=xb_test,
                         Xmu_test=Xbt_mu,
                         image_shape=image_shape,
                         fname=f"recons_t{i+1}")

        ## ═════════════════════════ Save current state of synapses to disk  ═════════════════════════
        model.save_to_disk(params_only=True)    ## save final state of synapses to disk


## ═══════════════════ Show Synapses Statistics  ══════════════════════
print(model.get_synapse_stats())

## ═══════════════════ Save Model  ════════════════════════════════════
## collect a test sample raster plot
model.save_to_disk(params_only=True) ## save final model parameters to disk



