from jax import jit, random
import os
from ngclearn import Context, numpy as jnp
import numpy as np
from hierarchical_pc import HierarchicalPredictiveCoding
import sys, getopt as gopt, optparse, time
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from ngclearn.utils.patch_utils import generate_patch_set
import sympy


"""
################################################################################
Reconstructive Predictive Coding Exhibit File:

This mode is fit to learn latent representations of the input and reconstructs 
input data sampled from the natural_scenes database. 

Usage:
$ python train_rpc.py --path_data="/path/to/dataset_arrays/" 
                      --n_samples=-1 --n_iter=3

Note that there is an optional argument "--n_samples", which allows you to choose a
number less than your argument dataset's total size N for cases where you are
interested in only working with a subset of the first K samples, where K < N. 
Further note that this script assumes there is a training dataset array 
called `dataX.npy` within `path/to/dataset_arrays`..

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# ################################################################################
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["path_data=",
                                                    "n_samples=",
                                                    "n_iter="])
model_type = "/hpc_rao1999"
experiment_circuit_name = "ExtraClassicalRF"
dataset_name = "/natural_scenes"
path_data = "../../data" + dataset_name

exp_dir = "exp" + model_type + "/" + experiment_circuit_name + dataset_name

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
################################################################################

jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)
################################################################################
## load the data
x_train = jnp.load(os.path.join(path_data, "dataX.npy"))

image_size = x_train.shape[1]
image_H = image_W = int(jnp.sqrt(image_size))
image_shape = (image_H, image_W)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
shuffle = True
mb_size = 100

# ═══════════════════════════════════════════════════════════════════════════
# PATCH ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
patch_shape = (8, 8)                           # Image patch dimensions
step_shape = (3, 3)                            # Step size for patches

# ────────────────────────────────────────────
# Input Layer - Receptive Fields
n_inPatch = 9                                  # Number of input patches (9 = full image)
pin_size = patch_shape[0] * patch_shape[1]     # Input patch dimension (64)

# ────────────────────────────────────────────
# Layer 1 (h1)
n_p1 = 9                                       # Number of h1 patches/PE-modules
p1_size = 32                                   # h1 patch dimension

# ────────────────────────────────────────────
# Layer 2 (h2) - Top Level Processing Elements
n_p2 = 1                                       # Number of h2 patches/PE-modules
p2_size = 64                                   # h2 patch dimension

# ═══════════════════════════════════════════════════════════════════════════
# COMPUTED DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════
h2_dim = p2_size * n_p2                        # = 64  × 1  = 64
h1_dim = p1_size * n_p1                        # = 32  × 9  = 288
in_dim = pin_size * n_inPatch                  # = 64  × 9  = 576


# ═══════════════════════════════════════════════════════════════════════════
# ENERGY DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
T = 20                                    # Number of E-steps
dt = 1.0

################################################################################
r1_prior_type = "laplacian"
r2_prior_type = "laplacian"
act_fx = "tanh"
k1 = 0.05
tau_m = 1 / k1

k2 = 0.05
sigma = 1.      ## sigma of layer 0
sigma_td = 10.  ## sigma of layer 1

alpha_1 = 1
alpha_2 = 0.05
################################################################################
x_train = generate_patch_set(x_train, patch_shape, max_patches=None, center=True, step_size=step_shape[0])
################################################################################
n_h = (image_shape[0] - patch_shape[0]) // step_shape[0] + 1
n_w = (image_shape[1] - patch_shape[1]) // step_shape[1] + 1
tot_patch_per_image = n_h * n_w

print("Network Dimensions: {} >> {} >> {}".format(h2_dim, h1_dim, in_dim))
print("Network Dimensions: {}{} >> {}{} >> {}{}".format(n_p2, p2_size,
                                                        n_p1, p1_size,
                                                        n_inPatch, pin_size
                                                        ))
################################################################################
## initialize and compile the model with fixed hyper-parameters



model = HierarchicalPredictiveCoding(dkey,
                                     circuit_name=experiment_circuit_name,
                                     h2_dim=h2_dim, h1_dim=h1_dim, in_dim=in_dim,
                                     n_p2=n_p2, n_p1=n_p1, n_inPatch=n_inPatch,
                                     batch_size=mb_size,
                                     dt=dt, T=T,
                                     tau_m=tau_m,
                                     lr=k2,
                                     sigma_e1 = sigma_td, sigma_e0 = sigma,
                                     act_fx=act_fx,                            ## non-linear activation function
                                     r1_prior=(r1_prior_type, alpha_1),
                                     r2_prior=(r2_prior_type, alpha_2),
                                     synaptic_prior = ("ridge", 0.02),
                                     exp_dir = exp_dir, reset_exp_dir = False
                                     )


model.save_to_disk() # NOTE: Viet: save initial model parameters to disk, uncomment this line if we are loading a saved model
model.load_from_disk(exp_dir) # NOTE: Viet: uncomment this line and comment the above lines to load a saved model
print(model.get_synapse_stats())
model.viz_receptive_fields(patch_shape, stride_shape=step_shape, fname='erf_t0')
################################################################################
## begin simulation of the model using the loaded data
if n_samples > 0:
    x_train = x_train[:n_samples, :]

total_batches = x_train.shape[0] // (n_inPatch * mb_size)

for i in range(n_iter):
    n_seen = 0
    cumultive_loss = 0
    print("========= Iter {}/{} ========".format(i+1, n_iter))
    for nb in range(total_batches):
        # ═══════════════════════════════════════════════════════════
        # BATCH EXTRACTION
        # ═══════════════════════════════════════════════════════════
        start_idx = nb * (n_inPatch * mb_size)
        end_idx = (nb + 1) * (n_inPatch * mb_size)

        Xb = x_train[start_idx:end_idx, :]          # Extract batch
        Xb = Xb.reshape(mb_size, -1)            # Reshape to (mb_size, 784)

        # ═══════════════════════════════════════════════════════════
        # MODEL PROCESSING
        # ═══════════════════════════════════════════════════════════

        Xmu, Lb = model.process(Xb, adapt_synapses=True)

        # ═══════════════════════════════════════════════════════════
        # METRICS UPDATE
        # ═══════════════════════════════════════════════════════════
        n_seen += Xb.shape[0]  # Total patterns seen
        cumultive_loss += Lb  # Accumulate loss
        avg_loss = cumultive_loss / (n_seen + 1)

        # ═══════════════════════════════════════════════════════════
        # PROGRESS DISPLAY
        # ═══════════════════════════════════════════════════════════
        print(
            f"\r "
            f"│ Seen: {n_seen:>6} patterns "
            f"│ Batch: {nb + 1:>4}/{total_batches:<4} "
            f"│ Train-Recon-Loss: {avg_loss:>7.4f} │",
            end=""
        )

    if (i+1) % iter_mod == 0:
        print()
        model.save_to_disk(params_only=True)                                ## save final state of synapses to disk
        model.viz_receptive_fields(patch_shape,
                                   stride_shape=step_shape,
                                   fname=f"erf_t{i+1}")


print(model.get_synapse_stats())

## collect a test sample raster plot
model.save_to_disk(params_only=True) ## save final model parameters to disk
