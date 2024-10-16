import pandas as pd
from jax import random, jit
import matplotlib.pyplot as plt
import numpy as np
from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.components import (HebbianSynapse,
                                 GaussianErrorCell)



class SparseDictLearning():
    def __init__(self, dkey, in_dim, h_dim, batch_size, opt_type="adam", fill=0.05, eta=0.001):
        
        dkey, *subkeys = random.split(dkey, 10)
        
        with Context("circuit") as self.circuit:
            self.W_lib = HebbianSynapse("W_lib", shape=(h_dim, in_dim), batch_size=batch_size, eta=eta,
                                   signVal=-1, sign_value=-1, weight_init=dist.constant(fill),
                                   optim_type=opt_type, key=subkeys[0])
            self.err = GaussianErrorCell("err", n_units=in_dim, batch_size=batch_size)
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.err.mu << self.W_lib.outputs
            self.W_lib.post << self.err.dmu
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            advance_cmd, advance_args = self.circuit.compile_by_key(self.W_lib, self.err, compile_key="advance_state")
            evolve_cmd, evolve_args = self.circuit.compile_by_key(self.W_lib, compile_key="evolve")
            reset_cmd, reset_args = self.circuit.compile_by_key(self.W_lib, self.err, compile_key="reset")
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.dynamic()

    def dynamic(self):  ## create dynamic commands for circuit
        W_lib, err = self.circuit.get_components("W_lib", "err")
        self.self = W_lib
        self.err = err

        @Context.dynamicCommand
        def clamp_inputs(targ, codes):
            self.W_lib.inputs.set(codes)
            self.W_lib.pre.set(codes)
            self.err.target.set(targ)

        self.circuit.wrap_and_add_command(jit(self.circuit.evolve), name="evolve")
        self.circuit.wrap_and_add_command(jit(self.circuit.advance_state), name="advance_state")
        self.circuit.wrap_and_add_command(jit(self.circuit.reset), name="reset")


    def sparsify(self, threshold=0.012, scale=1., T=1, dt=1.):
        W = self.W_lib.weights.value
        self.W_lib.weights.set(jnp.where(jnp.abs(W) <= threshold, 0., W * scale))
        self.circuit.advance_state(t=T * dt, dt=dt)
        return self.W_lib.weights.value


    def get_coeff(self, code_names, idx_names):
        coeff_ = self.sparsify()
        res_idx = jnp.any(coeff_ != 0, axis=1)
        res_coeff = coeff_[res_idx]
        res_names = [s for s, m in zip(code_names, res_idx) if m]
        print(pd.DataFrame(res_coeff, columns=res_names, index=idx_names))

    def process(self, target, z_codes, T=1, dt=1.):
        self.circuit.reset()
        self.circuit.clamp_inputs(targ=target, codes=z_codes)
        self.circuit.advance_state(t=T * dt, dt=dt)
        self.circuit.evolve(t=T, dt=dt)
        lib_coeff = np.array(self.W_lib.weights.value)

        return lib_coeff, self.err.mu.value, self.err.L.value




if __name__ == "__main__":
    from ngclearn.utils.diffeq.ode_functions import CreateLibrary
    from ngclearn.utils.diffeq.ode_functions import linear_2D as dfx
    from ngclearn.utils.diffeq.ode_utils_scanner import solve_ode
    key = random.PRNGKey(1234)

    x0 = jnp.array([3, -1.5])
    dt = 1e-2
    t0 = 0.
    T = 800

    ts, X = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=True)
    # -------------------------------------------Numerical derivate calculation---------------------------------------------
    scale = 2
    dX = jnp.array(np.gradient(jnp.array(X), ts, axis=0))
    dX = dX / scale
    # ------------------------------------------------codes---------------------------------------------
    Z_codes, code_names = CreateLibrary.poly_2D(X[:, 0], X[:, 1], deg=2, include_bias=True)  # N: datapoint, M: library features
    print('polynomial library (degree=2) \n')
    # ---------------------------------------------------------------------------------------------------
    batch_size = dX.shape[0]
    in_dim = dX.shape[1]
    h_dim = Z_codes.shape[1]
    dx_list = []
    model = sparse_dict_learn(dkey=key, in_dim=in_dim, h_dim=h_dim, batch_size=batch_size)
    for i in range(20000):
        coeff, d_xpred, efe_2 = model.process(target=dX, z_codes=Z_codes)

    sc_coeff = model.sparsify(scale=scale)
    model.get_coeff(code_names=code_names, idx_names=['dx', 'dy'])

    # print(pd.DataFrame(sc_coeff.T, columns=code_name, index=['dx', 'dy']))



