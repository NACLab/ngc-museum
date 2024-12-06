import jax
import pandas as pd
from jax import random, jit
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ngcsimlib.utils import Get_Compartment_Batch
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.components import (RateCell,
                                 HebbianSynapse,
                                 GaussianErrorCell,
                                 StaticSynapse)
from ngclearn.utils.model_utils import scanner


# batch = entire dataset --> time series?
# poly dict -> Dict(x) --> Dict(x-hat)

class itersindy():
    def __init__(self, key, out_dim, feat_dim, batch_size, fill=0.05, eta_2=0.01,
                 reg_type=None, w_decay=0.0,       #reg_type='l1', w_decay=0.0001,
                 T=1, dt=1., epochs=100):
        key, *subkeys = random.split(key, 10)

        self.T = T
        self.dt = dt
        self.epochs = epochs
        self.fill = fill

        feature_dim = feat_dim
        opt_type = "adam"

        with Context("circuit") as self.circuit:
            self.W_lib = HebbianSynapse("W_lib", shape=(feature_dim, out_dim), eta=eta_2,
                                   signVal=-1, sign_value=-1, weight_init=dist.constant(fill),
                                   optim_type=opt_type, reg_type=reg_type, w_decay=w_decay, key=subkeys[0])
            self.err = GaussianErrorCell("err", n_units=out_dim)

            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.W_lib.batch_size = batch_size
            self.err.batch_size = batch_size
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.err.mu << self.W_lib.outputs
            self.W_lib.post << self.err.dmu
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            advance_cmd, advance_args =self.circuit.compile_by_key(self.W_lib,  ## execute prediction synapses
                                                               self.err,  ## finally, execute error neurons
                                                               compile_key="advance_state")
            evolve_cmd, evolve_args =self.circuit.compile_by_key(self.W_lib, compile_key="evolve")
            reset_cmd, reset_args =self.circuit.compile_by_key(self.err, self.W_lib, compile_key="reset")
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.dynamic()

    def dynamic(self):  ## create dynamic commands forself.circuit
        W_lib, err = self.circuit.get_components("W_lib", "err")
        self.self = W_lib
        self.err = err

        @Context.dynamicCommand
        def batch_set(batch_size):
            self.W_lib.batch_size = batch_size
            self.err.batch_size = batch_size

        @Context.dynamicCommand
        def clamp_inputs(targ_scaled, lib):
            self.W_lib.inputs.set(lib)
            self.W_lib.pre.set(lib)
            self.err.target.set(targ_scaled)

        self.circuit.wrap_and_add_command(jit(self.circuit.evolve), name="evolve")
        self.circuit.wrap_and_add_command(jit(self.circuit.advance_state), name="advance")
        self.circuit.wrap_and_add_command(jit(self.circuit.reset), name="reset")


        # @scanner
        # def _process(compartment_values, args):
        #     _t, _dt = args
        #     compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
        #     compartment_values = self.circuit.evolve_state(compartment_values, t=_t, dt=_dt)
        #     return compartment_values, compartment_values[self.W_lib.weights.path]


        # @scanner
        # def backprop(compartment_values, args):
        #     _t, _dt = args
        #     compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
        #     compartment_values = self.circuit.evolve(compartment_values, t=_t, dt=_dt)
        #     return compartment_values, compartment_values[self.W_lib.weights.path]



    def sparsify(self, threshold=0.05, scale=2, T=2, dt=1.):
        coef_ = self.coef_ #self.W_lib.weights.value
        # print(':', coef_.T)
        # self.W_idx = jnp.where(jnp.abs(coef_ / scale) >= threshold, coef_ * scale, 0.)
        # lib_coeff = coef_ * self.W_idx
        # print('1 / scale)', (coef_ / scale).T)

        new_coeff = jnp.where(jnp.abs(coef_) >= threshold, coef_, 0.)
        # print('new_coeff', new_coeff.T)
        # print('new_coeff * scale', (new_coeff * scale).T)

        self.coef_ = new_coeff * scale
        self.W_lib.weights.set(new_coeff)
        # self.circuit.advance_state(t=T * dt, dt=dt)
        return self.coef_, self.W_lib.outputs.value# self.W_lib.weights.value, self.W_lib.outputs.value



    # @Context.dynamicCommand
    # def norm(self):
    #     self.W1.weights.set(normalize_matrix(self.W1.weights.value, wnorm=1., order=2, axis=1))
    #     self.W2.weights.set(normalize_matrix(self.W2.weights.value, wnorm=1., order=2, axis=1))
    #     self.W3.weights.set(normalize_matrix(self.W3.weights.value, wnorm=1., order=2, axis=1))



    def fit(self, dx, lib):

        self.circuit.reset()
        self.circuit.clamp_inputs(targ_scaled=dx, lib=lib)

        # z_codes = self.circuit._process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))

        gs1, gs2, gs3 = [], [], []
        mm = []
        for i in range(self.epochs):
            # gs1.append(self.W_lib.dWeights.value.ravel())
            self.circuit.advance(t=self.T * self.dt, dt=self.dt)
            gs2.append(self.W_lib.dWeights.value.ravel())
        #     # z_codes = self.circuit._process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))
            self.circuit.evolve(t=self.T, dt=self.dt)
            gs3.append(self.W_lib.dWeights.value.ravel())

            # print(i, (jnp.abs(jnp.array(gs3)[:, 0]) - self.fill).mean(),
            #       (jnp.abs(jnp.array(gs3)[:, 1]) - self.fill).mean(),
            #       (jnp.abs(jnp.array(gs3)[:, 2]) - self.fill).mean(),
            #       (jnp.abs(jnp.array(gs3)[:, 3]) - self.fill).mean(),
            #       (jnp.abs(jnp.array(gs3)[:, 4]) - self.fill).mean())


            # print(self.err.dmu.value.mean(),
            #       self.W_lib.dWeights.value[0],
            #       self.W_lib.dWeights.value[1],
            #       self.W_lib.dWeights.value[2],
            #       self.W_lib.dWeights.value[3],
            #       self.W_lib.dWeights.value[4])

            # mm.append(self.err.dmu.value)
        # print((jnp.abs(jnp.array(gs3)[:, 0]) - self.fill).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 1]) - self.fill).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 2]) - self.fill).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 3]) - self.fill).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 4]) - self.fill).mean())

        # print((jnp.abs(jnp.array(gs3)[:, 0])).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 1])).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 2])).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 3])).mean(),
        #       (jnp.abs(jnp.array(gs3)[:, 4])).mean())
        # input()

        # gs = gs1 + gs2 + gs3
        self.coef_ = np.array(self.W_lib.weights.value)

        return self.coef_, self.err.mu.value, self.err.L.value, gs2, gs3
