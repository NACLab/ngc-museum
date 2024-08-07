from ngclearn.utils.patch_utils import patch_with_stride
from patched_gpc import HierarchicalPatching_GPC
from jax import random
from ngclearn import numpy as jnp
import ngclearn.utils.weight_distribution as dist
import numpy as np



def gaussian_filter(shape, sigma):
    x_ = np.linspace(0, shape[0] - 1, shape[0])
    y_ = np.linspace(0, shape[1] - 1, shape[1])

    x, y = np.meshgrid(x_, y_)

    x_center = shape[0] // 2
    y_center = shape[1] // 2

    dist = jnp.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * (sigma ** 2)))
    return dist / np.sum(dist)

# ########################################################################
# ◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠◠
# datasets from http://www.rctn.org/bruno/sparsenet/
imgs = np.load('../../../dataset_Hierarchical_Rao/datasets/IMAGES.npy')
H, W , _ = imgs.shape
x_train = imgs.T
input_scale = 40.
# ◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡◡
# ########################################################################
n_iter = 200
viz_mod = 20
batch_mod = 50
n_samples = -1

k2 = 0.2      # eta
k1 = 1/35     # tau_m
gauss_sigma = 2.5 # 1.7

in_patchShape = (16, 16)
stride_shape = (5, 5)
mb_size = 100

n2, n1, n0 = (1, 3, 3)
d2, d1, d0 = (16, 8, in_patchShape[0] * in_patchShape[1])

z2_prior_type, z1_prior_type = ("cauchy", "cauchy")
alpha_td, alpha = (0.05, 1.)
sigma_td, sigma,  = (10, 1)
w_decay2, w_decay1 = (0.02, 0.06)

T = 100
dt = 1.

# ########################################################################
x_train = x_train[0:n_samples, :]

x_ = []
for idx in range(len(x_train)):
    _obs = x_train[idx, :]
    patched_obs = patch_with_stride(_obs, patch=in_patchShape, stride=stride_shape)
    x_.append(patched_obs)
x_ = jnp.array(x_)

gauss_filter = gaussian_filter(shape=in_patchShape, sigma=gauss_sigma)
x_ = x_[:, :x_.shape[1] - (x_.shape[1] % n0), :,:].reshape(-1, n0, 16, 16)
train_size, _, _, _ = x_.shape

x_gauss = np.array([(gauss_filter * x_[:, i, :]) for i in range(n0)]).reshape(train_size, -1)
x_train = jnp.array((x_gauss - np.mean(x_gauss)) * input_scale)

in_dim = in_patchShape[0] * in_patchShape[1] * n0
train_size = train_size//mb_size
x_train = x_train[:train_size * mb_size, :].reshape(train_size, mb_size, in_dim)

n_batches = x_train.shape[0]
# ########################################################################

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 12)

D2, D1, D0 = (n2 * d2, n1 * d1, n0 * d0)
eta1, eta2 = k2, k2
tau_m1, tau_m2 = 1/k1, 1/k1

model = HierarchicalPatching_GPC(dkey=subkeys[0],
                                 D2=D2, D1=D1, D0=D0,
                                 n0=n0, n1=n1, n2=n2,
                                 z2_prior_type=z2_prior_type, z2_lmbda_prior=alpha_td,
                                 z1_prior_type=z1_prior_type, z1_lmbda_prior=alpha,
                                 resist_scale1=1/sigma, resist_scale2=1/sigma_td,
                                 w_decay1=w_decay1, w_decay2=w_decay2,
                                 eta1=eta1, eta2=eta2,
                                 tau_m1=tau_m1, tau_m2=tau_m2,
                                 T=T, dt=dt, batch_size=mb_size,
                                 load_dir=None, exp_dir="exp")

print(model.get_synapse_stats())
model.save_to_disk()
model.viz_receptive_fields(rf1_shape=in_patchShape)

for iter in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], x_train.shape[0])
    X = x_train[ptrs, :]
    n_pat_seen = 0
    # print("\n ========= Iter {} ========".format(iter))
    L = 0.
    L1 = 0.
    
    if iter % 40 == 39:
        model.eta1 /= 1.015
        model.eta2 /= 1.015

    for idx in range(n_batches):
        Xb = X[idx, :]

        xs_mu, Lb = model.process(sensory_in=Xb, adapt_synapses=True)

        n_pat_seen += Xb.shape[0]
        L = Lb[0] + L
        L1 = Lb[1] + L1
        print("\r > Iteration {}    Seen {} patterns; Loss L1 = {}   ---   Loss L0{}".format(iter, n_pat_seen,
                                                                        (L1/mb_size), (L/mb_size)), end="")

        if (iter+1) % 10 == 0 and idx % batch_mod == 0 and idx > 0:
            print()
            model.viz_receptive_fields(rf1_shape=in_patchShape)
            model.save_to_disk(params_only=True)  # save final state of synapses to disk
            print(model.get_synapse_stats())
    print()
    if (iter+1) % viz_mod == 0:
        model.viz_receptive_fields(rf1_shape=in_patchShape)

## collect a test sample raster plot
model.save_to_disk(params_only=True)
