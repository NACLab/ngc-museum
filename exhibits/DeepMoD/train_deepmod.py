import jax
from jax import random, jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.integrate import solve_ivp

from iter_sindy import itersindy
from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
from ngclearn.utils.diffeq.odes import cubic_2D, linear_2D, lorenz, oscillator, linear_3D

from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.utils.diffeq.ode_solver import solve_ode
from ngclearn.components import (RateCell,
                                 HebbianSynapse,
                                 GaussianErrorCell,
                                 StaticSynapse)
from pc_pred import pc_predictor


# -------------------------------------
key = random.PRNGKey(1234)
key_ = random.PRNGKey(3476)
# # ------------------------------------------- System Configs  ---------------------------------------------
dfx = linear_2D
include_bias = False

if dfx == linear_2D:
    x0 = jnp.array([3, -1.5])
    deg = 2
    threshold = 0.01
    T = 800
    prob = 0.3
    w_fill = 0.05
    lr = 0.01
    include_bias = False
elif dfx == oscillator:
    x0 = jnp.array([-0.5, -0.05, -0.1])
    # x0 = jnp.array([0.1, 0.1, 0.1])
    deg = 2
    threshold = 0.005
    T = 1200
    include_bias = True
elif dfx == cubic_2D:
    x0 = jnp.array([2., 0.])
    deg = 3
    threshold = 0.05
    T = 1000
    # scale = 4           #scale = (dX.max() - dX.min()) / 4           # / 2 = / (max - min) = / (1 - (-1))
    prob = 0.3
    w_fill = 0.05
    lr = 0.001
    # p = prob/scale     # 0.05
    # inter = 1 + prob   # 1.3 / scale  == 1  / scale ===>
    include_bias = False
elif dfx == linear_3D:
    x0 = jnp.array([1, 1., -1])
    deg = 2
    threshold = 0.05
    T = 2000
    prob = 0.3
    lr = 0.001
    w_fill = 0.05
    include_bias = False
elif dfx == lorenz:
    x0 = jnp.array([-8, 8, 27], dtype=jnp.float32)
    threshold = 0.1
    deg = 2
    T = 500
    include_bias = False
# ----------------------------------    Solving System (for Data generation)     ------------------------------------------
n_epochs = 2000
dt = 1e-2
t0 = 0.

ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)

# -------------------------------------------Numerical derivate calculation---------------------------------------------
dX = jnp.array(np.gradient(jnp.array(X), jnp.array(ts), axis=0))

# ------------------------------------------- Create Library of features ---------------------------------------------
LibCreator = PolynomialLibrary(poly_order=deg, include_bias=include_bias)
feature_lib, feature_names = LibCreator.fit([X[:, i] for i in range(X.shape[1])])
# print(feature_names)
# print(jnp.std(feature_lib, axis=0), jnp.linalg.norm(feature_lib, axis=0))
# print(jnp.linalg.norm(feature_lib, axis=0) / jnp.linalg.norm(dX, axis=0))
# print(jnp.linalg.norm(feature_lib, axis=0) * jnp.linalg.norm(dX, axis=0))
# print(jnp.linalg.norm(dX, axis=0) / jnp.linalg.norm(feature_lib, axis=0))

# model = Std_SINDy(threshold=threshold, max_iter=100)
# print('---------- std-sindy coefficients ----------')
#
# dim_names = ['ẋ', 'ẏ', 'ż']
# preds = []
# loss = 0
# for i in range(X.shape[1]):
#     dx = dX[:, i:i+1]
#     sparse_coef = model.fit(dx=dx, lib=feature_lib)
#     pred = model.predict() #feature_lib @ sparse_coef
#     ode_ = model.get_ode(feature_names)
#
#     print(dim_names[i] + ' = ', *ode_)
#     loss += model.error()
#     preds.append(pred[:, 0])
#     # model.vis_sys(ts, dX, pred, model=dfx)
#
#
# dX_pred = jnp.array(preds).T
# model.vis_sys(ts, dX, dX_pred, model=dfx)
# print('loss', loss)
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
w_fill = w_fill * (scale * 0.5)  # scale / inter
lr = lr * (scale * 0.5)  # 0.01   # scale / inter
threshold = (threshold / scale) * (1 + prob)

# ##################################################################################################
# #                                  Predicting System
# ##################################################################################################
in_dim = ts_scaled.shape[1]
h1_dim = 16
h2_dim = 16
out_dim = X.shape[1]
batch_size = X.shape[0]
feat_dim = feature_lib.shape[1]


predictor = pc_predictor(key=key, in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, out_dim=out_dim,
                  batch_size=batch_size, T=10, eta=1e-2)

SReg_sindy = itersindy(key=key,out_dim=out_dim, feat_dim=feat_dim, batch_size=batch_size,
                     eta_2=1e-3, reg_type='l1', w_decay=0.0001, epochs=10)

coeff_track = 0
for i in range(n_epochs):
    x_pred, _, loss, _, _, _ = predictor.process(ts=ts_scaled, X=X, ad_dt=ts[:, None], scale=scale)

    dx_pred = jnp.array(np.gradient(jnp.array(x_pred), jnp.array(ts), axis=0))
    coeff, _, _, _, _ = SReg_sindy.fit(dx=dx_pred/scale, lib=feature_lib)

    cov_cria = (coeff_track - coeff).mean()
    coeff_track = coeff

    if jnp.abs(cov_cria) <= 5e-8 or i==n_epochs-1:
        coefficients = SReg_sindy.sparsify(threshold=threshold, scale=scale)[0]
        print('model converget at', i, 'with coefficients \n', coefficients.T)
        break



print('done')



