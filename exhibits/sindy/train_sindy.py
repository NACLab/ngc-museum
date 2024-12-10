import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ngclearn import Context, numpy as jnp
from sindy import Std_SINDy
from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary as FunctLib
from ngclearn.utils.diffeq.ode_solver import solve_ode
from ngclearn.utils.diffeq.odes import (linear_2D, cubic_2D,
                                        linear_3D, lorenz)

dfx = cubic_2D
include_bias = False

if dfx == linear_2D:
    x0 = jnp.array([3, -1.5])
    deg = 2
    threshold = 0.01
    T = 2000
    include_bias = False
elif dfx == cubic_2D:
    x0 = jnp.array([2., 0.])
    deg = 3
    threshold = 0.01
    T = 1000
    include_bias = False
elif dfx == linear_3D:
    x0 = jnp.array([1, 1., -1])
    deg = 2
    threshold = 0.001
    T = 2000
    include_bias = False
elif dfx == lorenz:
    x0 = jnp.array([-8, 8, 27], dtype=jnp.float32)
    threshold = 0.1
    deg = 2
    T = 500
    include_bias = False



dt = 1e-2 ## integration time constant (ms)
t0 = 0. ## initial condition time
ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)
# -------------------------------------------Numerical derivate calculation---------------------------------------------
dX = jnp.array(np.gradient(jnp.array(X), ts.ravel(), axis=0))
# ------------------------------------------------codes---------------------------------------------
lib_creator = FunctLib(poly_order=deg, include_bias=include_bias)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

model = Std_SINDy(threshold=0.1, max_iter=20)
print('---------- std-sindy coefficients ----------')
sparse_coef = model.fit(dx=dX, lib=feature_lib) ## fit sindy model to data
pred = model.predict() #feature_lib @ sparse_coef

ode_ = model.get_ode(feature_names)
c = jnp.where(sparse_coef==0, False, True)
idx_ = jnp.any(c == True, axis=1)
c_ = sparse_coef[idx_]
n_ = [name_ for name_, i_ in zip(feature_names, idx_) if i_]


plt.figure(facecolor='floralwhite')

if sparse_coef.shape[1]==3:
    df = pd.DataFrame(jnp.round(sparse_coef, 3), index=feature_names, columns=['ẋ', 'ẏ', 'ż'])
    plt.plot(ts, dX[:, 0], label=r'$\dot{x}$', linewidth=5, alpha=0.3, color='turquoise')
    plt.plot(ts, pred[:, 0], label=r'$\hat{\dot{x}}$', linewidth=0.8, ls="--", color='black')
    plt.plot(ts, dX[:, 1], label=r'$\dot{y}$', linewidth=4, alpha=0.6, color='pink')
    plt.plot(ts, pred[:, 1], label=r'$\hat{\dot{y}}$', linewidth=0.8, ls='--', color='red')
    plt.plot(ts, dX[:, 2], label=r'$\dot{z}$', linewidth=2, alpha=0.8, color='yellow')
    plt.plot(ts, pred[:, 2], label=r'$\hat{\dot{z}}$', linewidth=0.8, ls="--", color='navy')

elif sparse_coef.shape[1]==2:
    df = pd.DataFrame(jnp.round(sparse_coef, 3), index=feature_names, columns=['ẋ', 'ẏ'])
    plt.plot(ts, dX[:, 0], label=r'$\dot{x}$', linewidth=5, alpha=0.3, color='turquoise')
    plt.plot(ts, pred[:, 0], label=r'$\hat{\dot{x}}$', linewidth=0.8, ls="--", color='black')
    plt.plot(ts, dX[:, 1], label=r'$\dot{y}$', linewidth=4, alpha=0.6, color='pink')
    plt.plot(ts, pred[:, 1], label=r'$\hat{\dot{y}}$', linewidth=0.8, ls='--', color='red')

print(ode_) ## print out learned model statistics

plt.grid(True)
plt.legend(loc='lower right')
plt.xlabel(r'$time$', fontsize=10)
plt.ylabel(r'$\{\dot{x}, \dot{y}, \dot{z}\}$', fontsize=8)

plt.title('Sparse Coefficients of {} model'.format(str(dfx.__name__)))
#plt.savefig("model_fit_plot.png")
