from ngclearn.utils.patch_utils import generate_patch_set
from ngclearn.utils.patch_utils import Create_Patches, patch_with_stride
from jax import numpy as jnp, random
import sys, getopt as gopt, optparse, time
from patched_gpc import HierarchicalPatching_GPC
from jax import random, jit
from scipy.ndimage import gaussian_filter
import cv2
from ngclearn import numpy as jnp
import matplotlib.pyplot as plt
import jax.numpy as jnp
import ngclearn.utils.weight_distribution as dist
import numpy as np
import cv2
import matplotlib.patches as patches
from tqdm.notebook import tqdm
import scipy.io as sio

# DoG filter as a model of LGN
def DoG(img, ksize=(5, 5), sigma=1.3, k=1.6):
    g1 = cv2.GaussianBlur(img, ksize, sigma)
    g2 = cv2.GaussianBlur(img, ksize, k * sigma)
    dog = g1 - g2
    return (dog - dog.min()) / (dog.max() - dog.min())

# Gaussian mask for inputs
def GaussianMask(shape, sigma=1.7):
    sizex, sizey = shape
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    x0 = sizex // 2
    y0 = sizey // 2
    mask = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    return mask / np.sum(mask)



num_images = 10
num_iter = 5000


# ◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠
## datasets from http://www.rctn.org/bruno/sparsenet/
imgs = np.load('../../../dataset_Hierarchical_Rao/datasets/IMAGES.npy')
# Get image from imglist
H, W , n_samples = imgs.shape
x_train = imgs.T
input_scale = 40.
# ◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡

# ◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠
# # X = jnp.load('../../data/natural_scenes/dataX.npy').reshape(-1, 512, 512)
#
# X0 = jnp.load('../../../dataset_Hierarchical_Rao/image0.npy').reshape(-1, 408, 512)
# X1 = jnp.load('../../../dataset_Hierarchical_Rao/image1.npy').reshape(-1, 408, 512)
# X2 = jnp.load('../../../dataset_Hierarchical_Rao/image2.npy').reshape(-1, 408, 512)
# X3 = jnp.load('../../../dataset_Hierarchical_Rao/image3.npy').reshape(-1, 408, 512)
# X4 = jnp.load('../../../dataset_Hierarchical_Rao/image4.npy').reshape(-1, 408, 512)
# x_train = jnp.concatenate([X0, X1, X2, X3, X4])
# n_samples, H, W = x_train.shape
# ◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡

# ◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠
# n_samples = 5000
# X = jnp.load('../../data/mnist/trainX.npy').reshape(-1, 28, 28)
# x_train = X[0:n_samples, :]
# ◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠
dt = 1.
T = 100
n_iter = 100

k2 = 0.2
k1 = 1/30   # tau_m

in_patchShape = (16, 16)
stride_shape = (5, 5)

n0 = 3
n2, n1, n0 = (1, n0, n0)
d2, d1, d0 = (16, 8, in_patchShape[0] * in_patchShape[1])

act_fx2, act_fx1 = ("identity", "identity")
z2_prior_type, z1_prior_type = ("cauchy", "cauchy")
alpha_td, alpha = (0.05, 1.)
sigma_td, sigma,  = (10, 1)
w_decay2, w_decay1 = (0.02, 0.06)
# w1_init, w2_init = dist.fan_in_gaussian(), dist.fan_in_gaussian()
w1_init, w2_init = dist.gaussian(0., 1/5), dist.gaussian(0., 1/5)

# ########################################################################
x_ = []
for idx in range(len(x_train)):
    _obs = x_train[idx, :]
    patched_obs = patch_with_stride(_obs, patch=in_patchShape, stride=stride_shape)
    x_.append(patched_obs)
x_ = jnp.array(x_)

gaussian_mask = GaussianMask(shape=in_patchShape)
x_ = x_[:, :x_.shape[1] - (x_.shape[1] % n0), :,:].reshape(-1, n0, 16, 16)
train_size, _, _, _ = x_.shape

x_gauss = np.array([(gaussian_mask * x_[:, i, :]) for i in range(n0)]).reshape(train_size, -1)

x_train = jnp.array((x_gauss - np.mean(x_gauss)) * input_scale)

# ########################################################################

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 12)


D2, D1, D0 = (n2 * d2, n1 * d1, n0 * d0)
eta1, eta2 = k2, k2
tau_m1, tau_m2 = 1/k1, 1/k1

model = HierarchicalPatching_GPC(dkey=subkeys[0],
                                 D2=D2, D1=D1, D0=D0,
                                 n0=n0, n1=n1, n2=n2,
                                 weight1_init=w1_init, weight2_init=w2_init, bias_init=None,
                                 z2_prior_type=z2_prior_type, z2_lmbda_prior=alpha_td,
                                 z1_prior_type=z1_prior_type, z1_lmbda_prior=alpha,
                                 resist_scale1=1/sigma, resist_scale2=1/sigma_td,
                                 w_decay1=w_decay1, w_decay2=w_decay2,
                                 optim_type1="sgd", optim_type2="sgd",
                                 act_fx2=act_fx2, act_fx1=act_fx1,
                                 eta1=eta1, eta2=eta2,
                                 tau_m1=tau_m1, tau_m2=tau_m2,
                                 T=T, dt=dt, batch_size=1,
                                 load_dir=None, exp_dir="exp")


model.save_to_disk()
print(model.get_synapse_stats())
model.viz_receptive_fields(rf1_shape=in_patchShape)

for iter in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], x_train.shape[0])
    X = x_train[ptrs, :]
    n_pat_seen = 0
    print("\n ========= Iter {} ========".format(iter))
    L = 0.
    L1 = 0.
    if iter % 40 == 39:
        model.eta1 /= 1.015
        model.eta2 /= 1.015

    for idx in range(train_size):
        Xb = X[idx: idx + 1, :].reshape(1, -1)
        xs_mu, Lb = model.process(sensory_in=Xb, adapt_synapses=True)
        n_pat_seen += Xb.shape[0]
        L = Lb[0] + L
        L1 = Lb[1] + L1

        print("\r > Seen {} patterns; L = {}---{}              eta:{}".format(n_pat_seen,
                                                                        (L1/(idx+1) * 1.), (L/(idx+1) * 1.),
                                                                           [model.eta1, model.eta2]), end="")
    if iter % 5 == 0 and idx==train_size-1:
        print()
        model.viz_receptive_fields(rf1_shape=in_patchShape)
        model.save_to_disk(params_only=True)

## collect a test sample raster plot
model.save_to_disk(params_only=True)