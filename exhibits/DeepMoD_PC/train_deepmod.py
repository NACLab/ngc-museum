import jax
from jax import random, jit
import numpy as np
from ngclearn import Context
import jax.numpy as jnp
# print(jax.__version__)
from deepmod import DeepMoD
from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
from ngclearn.utils.diffeq.odes import cubic_2D, linear_2D, lorenz, oscillator, linear_3D
from ngclearn.utils.diffeq.ode_solver import solve_ode
# -------------------------------------
np.set_printoptions(suppress=True, precision=3)

key = random.PRNGKey(1234)
key_ = random.PRNGKey(3476)
# # ------------------------------------------- System Configs  ---------------------------------------------
dfx = linear_2D
include_bias = False
eta = 0.01

if dfx == linear_2D:
    x0 = jnp.array([3, -1.5], dtype=jnp.float32)
    deg = 2
    threshold = 0.02
    T = 800
    prob = 0.3
    w_fill = 0.05
    lr = 0.01
    include_bias = False
elif dfx == linear_3D:
    x0 = jnp.array([1, 1., -1], dtype=jnp.float32)
    deg = 2
    threshold = 0.05
    T = 2000
    prob = 0.3
    lr = 0.01
    w_fill = 0.05
    include_bias = False
elif dfx == cubic_2D:
    x0 = jnp.array([2., 0.], dtype=jnp.float32)
    deg = 3
    threshold = 0.05
    T = 1000
    # scale = 4           #scale = (dX.max() - dX.min()) / 4           # / 2 = / (max - min) = / (1 - (-1))
    w_fill = 0.05
    lr = 0.01
    prob = 0.3
    # p = prob/scale     # 0.05
    # inter = 1 + prob   # 1.3 / scale  == 1  / scale ===>
    include_bias = False
elif dfx == lorenz:
    x0 = jnp.array([-8, 8, 27], dtype=jnp.float32)
    threshold = 0.5
    deg = 2
    T = 1000
    prob = 0.3
    eta = 0.02
    w_fill = 0.05
    lr = 0.002
    include_bias = False
    # scale = 4

# ----------------------------------    Solving System (for Data generation)     ------------------------------------------
n_epochs = 2000
dt = 1e-2
t0 = 0.

ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)
# -------------------------------------------Numerical derivate calculation---------------------------------------------
dX = jnp.array(np.gradient(jnp.array(X), jnp.array(ts), axis=0))

# ------------------------------------------- Create Library of features ---------------------------------------------
library_creator = PolynomialLibrary(poly_order=deg, include_bias=include_bias)
feature_lib, feature_names = library_creator.fit([X[:, i] for i in range(X.shape[1])])

# --------------------------------------------  Preprocessing  -------------------------------------------
min = -1
max = 1
new_rng = max - min
t_min, t_max = ts.min(), ts.max()
data_rng = ts.max() - ts.min()                   # ts_scaled ~ [-1, 1]  - shape: (800, 1)
scale_ = data_rng / new_rng
ts_shifted = ts - t_min
ts_1centered = min + (ts_shifted / scale_)
ts_scaled = ts_1centered.reshape(ts.shape[0], 1)

scale = scale_ / 2
w_fill = w_fill * (scale / 2)  # scale / inter
lr = lr * (scale * 0.5)  # 0.01   # scale / inter
threshold = (threshold / scale) * (1 + prob)
# threshold = threshold / scale
# ##################################################################################################
# #                                              System
# ##################################################################################################

in_dim = ts_scaled.shape[1]
h1_dim = 16
h2_dim = 16
out_dim = X.shape[1]
batch_size = X.shape[0]
feat_dim = feature_lib.shape[1]
lasso_lmbda = 0.

deepmod = DeepMoD(key=key, ts=ts[:, None], dict_dim=feat_dim, lib_creator=library_creator,
                  solver_name="l1", l1_ratio=0.5, eta=eta,
                  in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim,
                  batch_size=batch_size, threshold=threshold, scale=scale,
                  w_fill=w_fill, lr=lr, lmbda=lasso_lmbda)


coeff_track = 0
for i in range(n_epochs):
    coeff, loss_pred = deepmod.process(ts_scaled, X)

    print("\r >epoch={}  L= {:.4f}| Sparse Weight: Wdx = {} | Wdy = {}".format(i,
                                                                     loss_pred/T,
                                                                     (deepmod.thresholding() * scale).T[0],
                                                                     (deepmod.thresholding() * scale).T[1]), end="")
    if i%100 == 0:
        print()

    cov_cria = (coeff_track - coeff).mean()
    coeff_track = coeff
    if jnp.abs(cov_cria) <= 5e-8:
        print('model converget at', i, 'with coefficients \n', deepmod.thresholding().T)
        break
print()



print('done')



