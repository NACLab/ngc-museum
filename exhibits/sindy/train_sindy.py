import numpy as np
from ngclearn import Context, numpy as jnp
from sindy import Std_SINDy
from ngclearn.utils.diffeq.feature_library import PolynomialLibrary
from ngclearn.utils.diffeq.ode_utils_scanner import solve_ode
from ngclearn.utils.diffeq.odes import (linear_2D, cubic_2D,
                                        linear_3D, lorenz, oscillator)

dfx = oscillator
include_bias = False

if dfx == linear_2D:
    x0 = jnp.array([3, -1.5])
    deg = 2
    threshold = 0.01
    T = 2000
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



dt = 1e-2
t0 = 0.
ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)
# -------------------------------------------Numerical derivate calculation---------------------------------------------
dX = jnp.array(np.gradient(jnp.array(X), ts.ravel(), axis=0))
# ------------------------------------------------codes---------------------------------------------
lib_creator = PolynomialLibrary(poly_order=deg, include_bias=include_bias)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

model = Std_SINDy(threshold=threshold, max_iter=100)
print('---------- std-sindy coefficients ----------')

dim_names = ['ẋ', 'ẏ', 'ż']
preds = []
for i in range(X.shape[1]):
    dx = dX[:, i:i+1]
    sparse_coef = model.fit(dx=dx, lib=feature_lib)
    pred = model.predict() #feature_lib @ sparse_coef
    ode_ = model.get_ode(feature_names)

    print(dim_names[i] + ' = ', *ode_)
    preds.append(pred[:, 0])

dX_pred = jnp.array(preds).T
model.vis_sys(ts, dX, dX_pred, model=dfx)
