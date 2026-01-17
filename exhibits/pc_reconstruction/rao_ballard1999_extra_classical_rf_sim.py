from jax import jit, random
import os
from ngclearn import numpy as jnp
from hierarchical_pc import HierarchicalPredictiveCoding
import sys, getopt as gopt, optparse, time
from ngclearn.components.input_encoders.ganglionCell import create_patches


"""
################################################################################
Hierarchical Predictive Coding for Extra-Classical Receptive Field Effects 
Exhibit File:

This mode is fit to learn localized, oriented Gabor-like receptive fields 
for context-dependent surround modulation from the natural_scenes database.

Usage:
$ python rao_ballard1999_extra_classical_rf_sim.py" 

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# ################################################################################
jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)

# ═══════════════════════════════════════════════════════════════════════════
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["n_iter="])

experiment_circuit_name = "Rao1999_ExtraClassicalRF"
dataset_name = "/natural_scenes"
path_data = "../../data" + dataset_name

exp_dir = "exp/" + experiment_circuit_name + dataset_name

n_samples = -1
n_iter = 10                         ## total number passes through dataset
iter_mod = 2

for opt, arg in options:
    if opt in ("--n_iter"):
        n_iter = int(arg.strip())

print("Data Path: ", path_data)

# ═══════════════════════════════════════════════════════════════════════════
## load the data
images = jnp.load(os.path.join(path_data, "dataX.npy"))

image_size = images.shape[1]
image_H = image_W = int(jnp.sqrt(image_size))
image_shape = (image_H, image_W)

images = images.reshape(-1, *image_shape)
# ═══════════════════════════════════════════════════════════════════════════
# Training Configuration
shuffle = True
mb_size = 100

# ════  Stimuli Configuration  ══════════════════════════════════════════════
area_shape = (14, 14)                          ## higher level receptive field shape
patch_shape = (8, 8)                           ## level-1 receptive fields shape
step_shape = (3, 3)                            ## level-1 receptive fields distance

# ──────────────── Input Layer - Receptive Fields
n_inPatch = 9                                  ## Number of level-1 PE-modules
pin_size = patch_shape[0] * patch_shape[1]     # Input patch dimension (64) = px * py = 8 * 8

# ──────────────── Layer 1 (h1) - middle Level Processing Elements
n_p1 = 9                                       ## Number of h1 patches/PE-modules
p1_size = 32                                   ## Size of each level-1 PE-module

# ─────────────── Layer 2 (h2) - Top Level Processing Elements
n_p2 = 1                                       ## Number of level-2 PE-modules
p2_size = 64                                   ## Size of each level-2 PE-module

## ═══════════════════════════════════════════════════════════════════════════
## Computed Dimensions
h2_dim = p2_size * n_p2                        # = 64  × 1  = 64
h1_dim = p1_size * n_p1                        # = 32  × 9  = 288
in_dim = pin_size * n_inPatch                  # = 64  × 9  = 576


# ═══════════════════════════════════════════════════════════════════════════
act_fx = "tanh"

r1_prior_type, alpha_1 = ("laplacian", 1)
r2_prior_type, alpha_2 = ("laplacian", 0.05)

k1 = 0.05
k2 = 0.05

sigma = 1.                  ## sigma of layer 0
sigma_td = 10.              ## sigma of layer 1

# ═══════════════════════════════════════════════════════════════════════════
# Energy Dynamics
T = 30                                    # Number of E-steps
dt = 1.
tau_m = 1 / k1

################################################################################
## initialize and compile the model with fixed hyper-parameters
model = HierarchicalPredictiveCoding(dkey,
                                     circuit_name = experiment_circuit_name,
                                     h2_dim=h2_dim, h1_dim=h1_dim, in_dim=in_dim,
                                     n_p2=n_p2, n_p1=n_p1, n_inPatch=n_inPatch,
                                     area_shape = area_shape,
                                     patch_shape = patch_shape,
                                     step_shape = step_shape,
                                     batch_size = mb_size,
                                     dt=dt, T=T,
                                     act_fx=act_fx,                            ## non-linear activation function
                                     tau_m=tau_m,
                                     lr=k2,
                                     sigma_e1 = sigma_td, sigma_e0 = sigma,
                                     r1_prior=(r1_prior_type, alpha_1),
                                     r2_prior=(r2_prior_type, alpha_2),
                                     synaptic_prior = ("ridge", 0.02),
                                     exp_dir = exp_dir, reset_exp_dir = True
                                     )

model.save_to_disk()          # NOTE: save initial model parameters to disk, uncomment this line if we are loading a saved model
model.load_from_disk(exp_dir) # NOTE: uncomment this line and comment the above lines to load a saved model
print(model.get_synapse_stats())
model.viz_receptive_fields(fname="erf_t0")

################################################################################
x_train = create_patches(images, patch_shape=area_shape, step_shape=area_shape)
x_train = x_train.reshape(-1, *area_shape)
total_batches = x_train.shape[0] // mb_size

for i in range(n_iter):
    n_seen = 0
    cumultive_loss = 0
    for nb in range(total_batches):
        # ════════════════════  get data batch  ═════════════════════
        mb = nb * mb_size
        Xb = x_train[mb:mb + mb_size, :]        # Extract batch: (B | 14, 14)

        # ═══════════════════ Model Processing ══════════════════════
        model.process(Xb, adapt_synapses=True)

        # ════════════════════   Metric update  ═════════════════════
        Lb = model.e0.L.get()                ## batch reconstruction loss
        n_seen += Xb.shape[0]                ## Total patterns seen
        cumultive_loss += Lb                 ## Accumulate loss
        avg_loss = cumultive_loss / (n_seen + 1)

        # ═══════════════════   Progress Display  ════════════════════
        print( f"\r "
            f"│ Seen: {n_seen:>6} patterns "
            f"│ Batch: {nb + 1:>4}/{total_batches:<4} "
            f"│ Train-Recon-Loss: {avg_loss:>7.4f} │",
            end=""
        )

    if (i+1) % iter_mod == 0:
        print()
        model.save_to_disk(params_only=True)                                ## save final state of synapses to disk
        model.viz_receptive_fields(fname=f"erf_t{(i+1) // iter_mod}")

print(model.get_synapse_stats())

## collect a test sample raster plot
model.save_to_disk(params_only=True) ## save final model parameters to disk
