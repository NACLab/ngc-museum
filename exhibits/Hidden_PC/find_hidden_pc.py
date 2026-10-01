import os
from jax import random
import sys, getopt as gopt, time
from ngclearn import numpy as jnp
from hidden_pc import HiddenPredictiveCoding
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.input_encoders.ganglionCell import _create_patches

"""
################################################################################
Hidden Predictive Coding (HiddenPC) Exhibit File:

This mode is fit to learn latent representations of the input and reconstructs 
input data sampled from the MNIST database. with fixed (random) synapses; only a
popup score per synapse is learned and each layer uses its top-k% synapses
(the edge-popup algorithm).

Usage:
$ python find_hidden_pc.py --model_name="sparsePC_patch" 
                           --path_data="/path/to/dataset_arrays/"                                                                                                                                                                                                                                               
                           --n_samples=-1 
                           --n_iter=10

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
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataset_name=",
                                                    "model_name="
                                                    "n_samples=",
                                                    "n_iter="])

experiment_circuit_name = "pc_patch"  ## pc_mlp, pc_patch, lateralPC_patch, sparsePC_patch, lateralPC_mlp, sparsePC_mlp,
dataset_name = "mnist"
path_data = "../../data/" + dataset_name

n_samples = -1
n_iter = 10        ## total number passes through dataset
iter_mod = 1

for opt, arg in options:
    if opt in ("--path_data"):
        path_data = arg.strip()
    elif opt in ("--model_name"):
        experiment_circuit_name = arg.strip()
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())

# ═══════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    "pc_mlp": dict(use_lateral=False, adaptive_lateral=False, exc_inh=(0., 0.), r_prior=(None, 0.),
                          area_shape=None, patch_shape=None, step_shape=None, w_prior=(None, 0.),
                          p3_size=16, p2_size=32*3, p1_size=32*9,
                          ),
    "pc_patch": dict(use_lateral=False, adaptive_lateral=False, exc_inh=(0., 0.), r_prior=(None, 0.),
                            area_shape=None, patch_shape=(14, 14), step_shape=(7, 7), w_prior=(None, 0.),
                            p3_size=16, p2_size=32, p1_size=64,
                            ),
    "lateralPC_mlp": dict(use_lateral=True, adaptive_lateral=True, exc_inh=(+5, -5), r_prior=(None, 0.),
                          area_shape=None, patch_shape=None, step_shape=None, w_prior=("ridge", 0.02),
                          p3_size=16, p2_size=32*3, p1_size=32*9,
                          ),
    "lateralPC_patch": dict(use_lateral=True, adaptive_lateral=True, exc_inh=(+5, -5), r_prior=(None, 0.),
                            area_shape=None, patch_shape=(14, 14), step_shape=(7, 7), w_prior=("ridge", 0.02),
                            p3_size=16, p2_size=32, p1_size=64,
                            ),
    "sparsePC_mlp": dict(use_lateral=False, adaptive_lateral=False, exc_inh=(0., 0.), r_prior=("laplacian", 0.14),
                         area_shape=None, patch_shape=None, step_shape=None, w_prior=("ridge", 0.02),
                         p3_size=16, p2_size=32*3, p1_size=32*9,
                         ),
    "sparsePC_patch": dict(use_lateral=False, adaptive_lateral=False, exc_inh=(0., 0.), r_prior=("laplacian", 0.14),
                           area_shape=None, patch_shape=(14, 14), step_shape=(7, 7), w_prior=("ridge", 0.02),
                           p3_size=16, p2_size=32, p1_size=64,
                           ),
}

pc_circuit = MODEL_CONFIGS[experiment_circuit_name]
exp_dir = "exp/" + experiment_circuit_name

# ═══════════════════════════════════════════════════════════════════════════
jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, n_iter + 10)
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
y_test = y_test[jnp.argsort(jnp.argmax(y_test, axis=1))]

image_size = img_train.shape[1]
ix = iy = int(jnp.sqrt(image_size))
image_shape = (ix, iy)

img_train = img_train.reshape(-1, *image_shape)
img_test = img_test.reshape(-1, *image_shape)

# ════  Stimuli Configuration  ══════════════════════════════════════════════
(ax, ay) = area_shape = pc_circuit["area_shape"] or image_shape  ## (ax, ay) = (ix, iy): full image
(px, py) = patch_shape = pc_circuit["patch_shape"] or image_shape  ## None == full image
(sx, sy) = step_shape = pc_circuit["step_shape"] or patch_shape  ## (sx, sy) --- ix = px + (nx-1) * sx

nx = (ax - px) // sx + 1 if sx > 0 else ax // px
ny = (ay - py) // sy + 1 if sy > 0 else ay // py

n_cells = nx * ny  ## ==1 means full image at the time image
n_p1 = nx * ny  ## number of h1 patches/PE-modules
n_p2 = ny  ## number of h2 patches/PE-modules
n_p3 = 1  ## number of h3 patches/PE-modules

p3_size = pc_circuit["p3_size"]  ## h3 patch dimension
p2_size = pc_circuit["p2_size"]  ## h2 patch dimension
p1_size = pc_circuit["p1_size"]  ## h1 patch dimension

pin_size = patch_shape[0] * patch_shape[1]  ## input patch dim (== h1 neurons receptive field size)

## ═══════════════════════════════════════════════════════════════════════════
## Computed Dimensions
h3_dim = p3_size * n_p3
h2_dim = p2_size * n_p2  ## = 128 × 1  = 128
h1_dim = p1_size * n_p1  ## =  32 × 3  = 96
in_dim = pin_size * n_cells  ## = 256 × 3  = 768

## ══════════════════════════════════════════════════════════════════════════
## Energy Dynamics
T = 30  ## number E-steps
dt = 1.
lr = 0.005
k=0.3

use_lateral = pc_circuit["use_lateral"]
adaptive_lateral = pc_circuit["adaptive_lateral"]
exc, inh = pc_circuit["exc_inh"]

## ══════════════════════════════════════════════════════════════════════════
## split the full image into local views for retinal ganglion cells local receptive fields
x_train = _create_patches(img_train, patch_shape=area_shape,
                          step_shape=area_shape)  ### shape: (N | n_areas | (area_shape))
x_test = _create_patches(img_test, patch_shape=area_shape,
                         step_shape=area_shape)  ### shape: (N | n_areas | (area_shape))

x_train = x_train.reshape(-1, *area_shape)  ### shape: (n_total_obs | (area_shape))
x_test = x_test.reshape(-1, *area_shape)  ### shape: (n_total_obs | (area_shape))

# ═══════════════════════════════════════════════════════════════════════════
################################################################################
## initialize and compile the model with fixed hyper-parameters
model = HiddenPredictiveCoding(dkey,
                                     circuit_name=experiment_circuit_name,
                                     h3_dim=h3_dim, h2_dim=h2_dim, h1_dim=h1_dim, in_dim=in_dim,
                                     n_p3=n_p3, n_p2=n_p2, n_p1=n_p1, n_inPatch=n_cells,
                                     area_shape=area_shape,
                                     patch_shape=patch_shape,
                                     step_shape=step_shape,
                                     batch_size=mb_size,
                                     T=T, dt=dt, tau_m=20, act_fx="relu",
                                     use_lateral=use_lateral, adaptive_lateral=adaptive_lateral,
                                     exc_inh=(exc, inh),
                                     lat_eta=lr if use_lateral else 0.,
                                     lr=lr, k=k,
                                     weight_init=dist.fan_in_signed_constant(gain=jnp.sqrt(2.)),
                                     r3_prior=pc_circuit["r_prior"],
                                     r2_prior=pc_circuit["r_prior"],
                                     r1_prior=pc_circuit["r_prior"],
                                     synaptic_prior=pc_circuit["w_prior"],
                                     exp_dir=exp_dir, reset_exp_dir=True
                                     )

model.save_to_disk()  # NOTE: save initial model parameters to disk, uncomment this line if we are loading a saved model
# model.load_from_disk(exp_dir) # NOTE: uncomment this line and comment the above lines to load a saved model
model.get_synapse_stats()
model.viz_receptive_fields(max_n_vis=mb_vis_size, fname='t0_[W]_[Score]_[W-Score]_[W-subnet]')

# ═══════════════════════════════════════════════════════════════════════════
## begin simulation of the model using the loaded data
if n_samples > 0:
    x_train = x_train[:n_samples, :]
    print("-> Fitting model to only {} samples".format(n_samples))

n_batch_train = x_train.shape[0] // mb_size
ptrs_ = random.permutation(subkeys[1], x_test.shape[0])
X_test = x_test[ptrs_, :]
Y_test = y_test[ptrs_, :]  ## labels follow the same shuffle (valid while area_shape == image_shape)

# ═══════════════════════ Latent Variable Analysis ══════════════════════════
Z_test_init, L_test = model.process_test(X_test)
######## Collect and Save Latents
model.collect_latents(Z_test_init, Y_test, save=True, fname="latents_init")
######## Effective Dimensionality
model.get_eff_dims(fname="latents_init")
######## Probe Accuracy
model.get_probe_acc(fname="latents_init")
######## t-SNE Visualization
model.plot_codes(fname="latents_init")
# ═══════════════════════════════════════════════════════════════════════════

start = time.time()
for i in range(n_iter):
    X = x_train
    # ════════════════════  shuffle   ═════════════════════
    if shuffle:
        ptrs = random.permutation(subkeys[0], x_train.shape[0])
        X = x_train[ptrs, :]

    # ═══════════════════════════════════════════════════════════════════════════
    n_seen = 0
    cumultive_loss = 0
    epoch_start = time.time()
    prev_subnet = None
    for nb in range(n_batch_train):
        # ════════════════════  get data batch  ═════════════════════
        mb = nb * mb_size
        Xb = X[mb:mb + mb_size, :]  # Extract batch: (B | ax, ay)

        # ═══════════════════ Model Processing ══════════════════════
        Xmu = model.process(Xb, adapt_synapses=True)

        # ════════════════════   Metric Update  ═════════════════════
        Lb = model.e0.L.get()  ## batch reconstruction loss
        cumultive_loss += Lb  ## Accumulate loss
        n_seen += Xb.shape[0]  ## Total patterns seen
        avg_loss = cumultive_loss / (n_seen + 1)

        # ═══════════════════   Progress Display  ════════════════════
        rate = n_seen / (time.time() - epoch_start)
        ##-------------------------------------------------------------------------------------------
        sub_network = jnp.concatenate([model.W1.subnet.get().ravel(),
                                model.W2.subnet.get().ravel(),
                                model.W3.subnet.get().ravel()])
        ## Turnover: Synapse formation (synaptogenesis) and synaptic pruning
        structural_plasticity = 0. if prev_subnet is None else float(
            jnp.sum(sub_network != prev_subnet) / 2 / (model.W1.n_keep + model.W2.n_keep + model.W3.n_keep))
        prev_subnet = sub_network
        ##-------------------------------------------------------------------------------------------

        print(f"\r "
              f"│ Iter: {i:>1} "
              f"│ Seen: {n_seen:>6} patterns "
              f"│ Batch: {nb + 1:>4}/{n_batch_train:<4} "
              f"│ {rate:>5.0f} patterns/s ",
              f"│ Kept: {k * 100:.0f}% "
              f"│ Structural-Plasticity: {structural_plasticity:.4f} "
              f"│ Train-Loss: {avg_loss:>7.4f} ",
              end="", flush=True
              )
    ###########################################################################
    if (i + 1) % iter_mod != 0:
        print()
    if (i + 1) % iter_mod == 0:
        ## ==============  L1 Synaptic Filters Display  =======================
        model.viz_receptive_fields(max_n_vis=mb_vis_size, fname=f"t{i + 1}_[W]_[Score]_[W-Score]_[W-subnet]")
        ## ==============  Save current state of synapses to disk  ============
        model.save_to_disk(params_only=True)

        ## =========================   TEST PHASE  ============================
        #############   infer test data
        Z_test, L_test = model.process_test(X_test)
        ######## Reconstruction Loss
        print(f"│ Test-Loss: {L_test :>7.4f}   │")

        ########   Save latent codes to disk
        model.collect_latents(Z_test, Y_test, save=True)
        ######## Effective Dimensionality
        model.get_eff_dims()
        ######## Probe Accuracy
        model.get_probe_acc()
        ######## t-SNE Visualization
        model.plot_codes()

        #############   infer and visualize test data
        xb_test = X_test[:mb_vis_size, :]
        xb_mu = model.process(xb_test, adapt_synapses=False)
        model.viz_recons(xb_test, xb_mu, image_shape,
                         image_area_step=image_shape if pc_circuit["patch_shape"] else (0, 0),
                         fname=f"recons_t{i + 1}")

print(f"\nTotal training time: {time.time() - start:.1f}s")
## ════════════════ Show Synapses Statistics  ════════════════
model.get_synapse_stats()

## ════════════════ Save Model  ══════════════════════════════
## save final model parameters to disc
model.save_to_disk(params_only=True)








