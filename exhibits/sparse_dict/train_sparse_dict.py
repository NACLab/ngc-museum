import pandas as pd
from jax import random, jit
import matplotlib.pyplot as plt
import numpy as np
from ngclearn import Context, numpy as jnp
from ngclearn.utils.diffeq.ode_functions import CreateLibrary
from ngclearn.utils.diffeq.ode_utils_scanner import solve_ode
from ngclearn.utils.diffeq.ode_functions import cubic_2D, linear_2D, lorenz_model
from sparse_dict import SparseDictLearning

key = random.PRNGKey(1234)



dfx = linear_2D

if dfx==linear_2D:
    x0 = jnp.array([3, -1.5])
    poly_deg = 2
elif dfx==cubic_2D:
    x0 = jnp.array([2., 0.])
    poly_deg = 3

dt = 1e-2
t0 = 0.
T = 800
n_epoch = 20000
scale = 2

ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)
# -------------------------------------------Numerical derivate calculation---------------------------------------------
dX = jnp.array(np.gradient(jnp.array(X), ts, axis=0))
dX = dX / scale
# ------------------------------------------------codes---------------------------------------------
Z_codes, code_names = CreateLibrary.poly_2D(X[:, 0], X[:, 1], deg=poly_deg, include_bias=False)  # N: datapoint, M: library features
print('polynomial library (degree=2)', '\n', code_names)
# ---------------------------------------------------------------------------------------------------
batch_size = dX.shape[0]
in_dim = dX.shape[1]
h_dim = Z_codes.shape[1]

model = SparseDictLearning(dkey=key, in_dim=in_dim, h_dim=h_dim, batch_size=batch_size)
# ---------------------------------------------------------------------------------------------------

dx_list = []
for i in range(n_epoch):
    coeff, d_xpred, efe = model.process(target=dX, z_code=Z_codes)

model.get_coeff(z_codes=Z_codes, target=dX, scale=scale)
model.get_coeff(z_codes=Z_codes, target=dX, scale=scale, code_names=code_names, idx_names=['dy', 'dx'])

plt.figure(facecolor='floralwhite')
plt.plot(ts, dX[:, 0], label=r'$\dot{x}$', linewidth=5, alpha=0.65, color='turquoise')
plt.plot(ts, d_xpred[:, 0], label=r'$\hat{\dot{x}}$', linewidth=2, linestyle="-.", color='navy')
plt.plot(ts, dX[:, 1], label=r'$\dot{y}$', linewidth=5, alpha=0.7, color='pink')
plt.plot(ts, d_xpred[:, 1], label=r'$\hat{\dot{y}}$', linewidth=2, linestyle='-.', color='red')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'$time$')
plt.ylabel(r'$\frac{d\hat{X}}{dt}$')
plt.title('Sparse Dictionary Coefficients of system: ' + str(dfx.__name__))
# plt.show()


