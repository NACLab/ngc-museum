from jax import jit, random
import os
from ngclearn import Context, numpy as jnp
import numpy as np
from pc_recon import PC_Recon
import sys, getopt as gopt, optparse, time
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from ngclearn.utils.patch_utils import generate_patch_set
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


# ################################################################################
## read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["path_data=",
                                                    "n_samples=",
                                                    "n_iter="])

path_data = "../../data/mnist/"
n_samples = -1
n_iter = 10                         ## total number passes through dataset
for opt, arg in options:
    if opt in ("--path_data"):
        path_data = arg.strip()
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
print("Data Path: ", path_data)

################################################################################
# load the data
jnp.set_printoptions(suppress=True, precision=5)
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 6)
################################################################################
x_train = jnp.load(os.path.join(path_data, "trainX.npy"))
x_test = jnp.load(os.path.join(path_data, "testX.npy"))

image_size = x_train.shape[1]
image_H = image_W = int(jnp.sqrt(image_size))
image_shape = (image_H, image_W)
################################################################################
T = 20 #30 # K = 100  ## number E-steps
dt = 1.
iter_mod = 1
viz_mod = 1

mb_size = 100
mb_vis_size = 100
max_vis_filter = 100

# patch_shape = image_shape    ## full image
patch_shape = (8, 8)           ## (image_shape) shape of each image patch
n_inPatch = 9                  ## ==1 means full image at the time image
n_p1 = 3                       ## number of h1 patches/PE-modules
n_p2 = 1                       ## number of h2 patches/PE-modules
n_p3 = 1                       ## number of h3 patches/PE-modules

p3_size = 32                   ## h3 patch dimension
p2_size = 64                   ## h2 patch dimension
p1_size = 128                  ## h1 patch dimension
pin_size = patch_shape[0] * patch_shape[1]     ## input patch dim (== h1 neurons receptive field size)

h3_dim, h2_dim, h1_dim, in_dim = (p3_size * n_p3,
                                  p2_size * n_p2,
                                  p1_size * n_p1,
                                  pin_size * n_inPatch
                                  )
################################################################################
n_h = image_shape[0] - patch_shape[0] + 1
n_w = image_shape[1] - patch_shape[1] + 1
tot_patch_per_image = n_h * n_w
print("tot_patch_per_image is ", tot_patch_per_image)
print("number of input patches is ", n_inPatch)
if tot_patch_per_image / n_inPatch != int(tot_patch_per_image / n_inPatch):
    input("recommended mb_size and n_inPatch", sympy.factorint(tot_patch_per_image, multiple=True))


# Returns a dictionary of prime factors and their multiplicities



if patch_shape == image_shape:  # Full image case
    images_per_batch = mb_size
else:  # Patch case
    n_samples = 1000              ## reduce computational cost
    mb_size = tot_patch_per_image // n_inPatch
    images_per_batch = 1


print("Network Dimensions: {} >> {} >> {} >> {}".format(h3_dim, h2_dim, h1_dim, in_dim))
print("Network Dimensions: {}{} >> {}{} >> {}{} >> {}{}".format(
                                                n_p3, p3_size,
                                                n_p2, p2_size,
                                                n_p1, p1_size,
                                                n_inPatch, pin_size,
))


print("    mini-batch size : ", mb_size)
print("    number of images per mini-batch: {}".format(mb_size))
print("    mini-batch size per image: ", images_per_batch)
################################################################################
## initialize and compile the model with fixed hyper-parameters
model = PC_Recon(dkey, h3_dim=h3_dim, h2_dim=h2_dim, h1_dim=h1_dim, in_dim=in_dim,
                 n_p3=n_p3,
                 n_p2=n_p2,
                 n_p1=n_p1,
                 n_inPatch=n_inPatch,
                 batch_size=mb_size,
                 circuit_name="train")

model.save_to_disk()
print(model.get_synapse_stats())
model.viz_receptive_fields(patch_shape, max_filter=max_vis_filter, fname='erf_t0')

################################################################################
## begin simulation of the model using the loaded data
if n_samples > 0:
    x_train = x_train[:n_samples, :]
    print("-> Fitting model to only {} samples".format(n_samples))

n_batches = x_train.shape[0]//images_per_batch

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 4)
    ptrs = random.permutation(subkeys[0], x_train.shape[0])
    ptrs_ = random.permutation(subkeys[1], x_test.shape[0])
    X = x_train[ptrs, :]
    X_test = x_test[ptrs_, :]

    n_pat_seen = 0
    L = 0
    print("========= Iter {}/{} ========".format(i+1, n_iter))
    for nb in range(n_batches):
        Xb = X[nb * images_per_batch: (nb + 1) * images_per_batch, :]                     # shape: (batch_size, 784)
        Xb = generate_patch_set(Xb, patch_shape, max_patches=None, center=True, seed=None)

        Xb = Xb.reshape(mb_size, -1)
        Xmu, Lb = model.process(Xb, adapt_synapses=True)

        n_pat_seen = n_pat_seen + Xb.shape[0]                               ## total number of patterns seen
        L = Lb + L                                                          ## track current global loss
        print("\r > Seen {} patterns   | n_batch {}/{}   | Train-Recon-Loss = {}".format(
            n_pat_seen, (nb+1), n_batches, L/(n_pat_seen+1)), end="")


    if i % iter_mod == 0:
        print()

        ##################   test phase
        Xtest = X_test[:mb_vis_size, :]

        ## for full image inputs
        if images_per_batch > 1:
            Xb_test, x_test_mean = generate_patch_set(Xtest, patch_shape, max_patches=None, seed=None,
                                                      center=True, vis_mode=True)
            ## only perform E-steps/inference
            Xt_mu, L_test = model.process(Xb_test.reshape(mb_size, -1), adapt_synapses=False)
            Xtest_mu = Xt_mu.reshape(mb_size * n_inPatch, -1) + x_test_mean
            ###############################################
            print("\r >  Test Recon Loss = {} ".format(L_test / mb_vis_size))
            model.viz_recons(Xtest, Xtest_mu, image_shape=image_shape, fname=f"recons_t{i+1}")

        ## for patched inputs
        elif images_per_batch == 1:
            l0 = 0
            Xmu_list = []
            for jb in range(mb_vis_size):
                Xt = Xtest[jb:jb+1, :]
                Xb_test, x_test_mean = generate_patch_set(Xt, patch_shape, max_patches=None, seed=None,
                                                          center=True, vis_mode=True)
                ## only perform E-steps/inference
                Xt_mu, L_test = model.process(Xb_test.reshape(mb_size, -1), adapt_synapses=False)
                l0 = l0 + L_test
                ## reconstruct images from patches
                Xtest_mu = Xt_mu.reshape(mb_size * n_inPatch, -1) + x_test_mean
                X_mu = reconstruct_from_patches_2d(np.asarray(Xtest_mu).reshape(-1, *patch_shape), image_shape)
                Xmu_list.append(X_mu.reshape(image_shape[0] * image_shape[1]))
            ###############################################
            print("\r  >  Test Recon Loss = {} ".format(l0 / (mb_vis_size * mb_size)))
            model.viz_recons(Xtest, np.asarray(Xmu_list), image_shape=image_shape, fname=f"recons_t{i+1}")

        ###############################################
        model.save_to_disk(params_only=True)                                    ## save final state of synapses to disk

    print()
    if i % viz_mod == 0:
        print(model.get_synapse_stats())
        model.viz_receptive_fields(patch_shape, max_filter=max_vis_filter, fname=f"erf_t{i+1}")

## collect a test sample raster plot
model.save_to_disk(params_only=True) ## save final model parameters to disk

