import pandas as pd
from jax import random, jit
import matplotlib.pyplot as plt
import numpy as np
from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.utils.model_utils import scanner
from ngclearn.components import (HebbianSynapse,
                                 GaussianErrorCell)
from ngclearn.utils.model_utils import normalize_matrix


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

    def normalize(self, lib_y, y):
        dict_norm = jnp.expand_dims(jnp.linalg.norm(lib_y, axis=0), axis=1)
        pred_norm = jnp.expand_dims(jnp.linalg.norm(y, axis=0), axis=0)
        coeff_norm = dict_norm * self.W_lib.weights.value / pred_norm
        return coeff_norm

    def sparsify(self, z_codes, target, scale=1., threshold=0.01):
        W_norm = self.circuit.normalize(z_codes, target)
        W_sparse = jnp.where(jnp.abs(W_norm) <= threshold, 0., W_norm)
        return W_sparse


    def get_coeff(self, z_codes, target, scale=1., code_names=None, idx_names=None, sparsify=True):
        if sparsify:
            coeff_ = self.sparsify(z_codes, target) * scale
        else:
            coeff_ = self.W_lib.weights.value * scale

        if not code_names:
            return print(pd.DataFrame(coeff_).T)

        res_idx = jnp.any(coeff_ != 0, axis=1)
        res_coeff = coeff_[res_idx]
        res_names = [s for s, m in zip(code_names, res_idx) if m]

        print(pd.DataFrame(res_coeff, columns=res_names, index=idx_names))
        

    def process(self, target, z_code):
        self.circuit.reset()
        self.circuit.clamp_inputs(targ=target, codes=z_code)
        self.circuit.advance_state(dt=1.)
        self.circuit.evolve()

        lib_coeff = np.array(self.W_lib.weights.value)
        return lib_coeff, self.err.mu.value, self.err.L.value
