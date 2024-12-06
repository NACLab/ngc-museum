import jax
import pandas as pd
from ngclearn import Component, Compartment, numpy as jnp, resolver
from jax import random, jit
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ngcsimlib.utils import Get_Compartment_Batch
from jax import grad
from jax import grad, jacfwd, jacrev
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

class pc_predictor():
    def __init__(self, key, in_dim, h1_dim, h2_dim, out_dim, batch_size,
                 eta = 1e-3, tau_m = 20., T=50, dt=1.):
        key, *subkeys = random.split(key, 10, )

        self.T = T
        self.dt = dt

        opt_type = "adam"
        act_fx = "sine"
        self.omega_0 = 30  # check 2-300-10

        W3_dist = dist.uniform(amin=-1 / h2_dim, amax=1 / h2_dim)
        W2_dist = dist.uniform(amin=-np.sqrt(6 / h1_dim) / self.omega_0,
                                    amax=np.sqrt(6 / h1_dim) / self.omega_0)
        W1_dist = dist.uniform(amin=-np.sqrt(6 / out_dim) / self.omega_0,
                                    amax=np.sqrt(6 / out_dim) / self.omega_0)

        with Context("model") as self.model:
            ############ L3
            self.z3 = RateCell("z3", n_units=in_dim, tau_m=tau_m , act_fx="identity")
            self.q3 = RateCell("q3", n_units=in_dim, tau_m=-1 , act_fx="identity")
            self.W3 = HebbianSynapse("W3", shape=(in_dim, h2_dim), eta=eta, signVal=-1, sign_value=-1,
                                     optim_type=opt_type, weight_init=W3_dist, key=subkeys[0])
            self.Q3 = StaticSynapse("Q3", shape=(in_dim, h2_dim))

            ############ L2
            self.e2 = GaussianErrorCell("e2", n_units=h2_dim)
            self.z2 = RateCell("z2", n_units=h2_dim, tau_m=tau_m , act_fx=act_fx, omega_0=self.omega_0)
            self.q2 = RateCell("q2", n_units=h2_dim, tau_m=-1, act_fx=act_fx, omega_0=self.omega_0)

            self.W2 = HebbianSynapse("W2", shape=(h2_dim, h1_dim), eta=eta, signVal=-1, sign_value=-1,
                                     optim_type=opt_type, weight_init=W2_dist, key=subkeys[1])
            self.Q2 = StaticSynapse("Q2", shape=(h2_dim, h1_dim))
            self.E2 = StaticSynapse("E2", shape=(h1_dim, h2_dim))

            ############ L1
            self.e1 = GaussianErrorCell("e1", n_units=h1_dim)
            self.z1 = RateCell("z1", n_units=h1_dim, tau_m=tau_m , act_fx="identity")
            self.q1 = RateCell("q1", n_units=h1_dim, tau_m=-1 , act_fx="identity")
            self.W1 = HebbianSynapse("W1", shape=(h1_dim, out_dim), eta=eta, signVal=-1, sign_value=-1, optim_type=opt_type,
                                weight_init=W1_dist, key=subkeys[2])
            self.Q1 = StaticSynapse("Q1", shape=(h1_dim, out_dim))
            self.E1 = StaticSynapse("E1", shape=(out_dim, h1_dim))

            ############ input
            self.e0 = GaussianErrorCell("e0", n_units=out_dim)

            # # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.z3.batch_size= batch_size
            self.z2.batch_size= batch_size
            self.z1.batch_size = batch_size



            self.e2.batch_size = batch_size
            self.e1.batch_size = batch_size
            self.e0.batch_size = batch_size

            self.W3.batch_size = batch_size
            self.W2.batch_size = batch_size
            self.W1.batch_size = batch_size

            self.E2.batch_size = batch_size
            self.E1.batch_size = batch_size

            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            self.W3.inputs << self.z3.zF
            self.e2.mu << self.W3.outputs

            self.e2.target << self.z2.z
            self.W2.inputs << self.z2.zF
            self.e1.mu << self.W2.outputs

            self.e1.target << self.z1.z
            self.W1.inputs << self.z1.zF
            self.e0.mu << self.W1.outputs

            self.z2.j_td << self.e2.dtarget
            self.E2.inputs << self.e1.dmu
            self.z2.j << self.E2.outputs

            self.z1.j_td << self.e1.dtarget
            self.E1.inputs << self.e0.dmu
            self.z1.j << self.E1.outputs

            self.W1.pre << self.z1.zF
            self.W1.post << self.e0.dmu

            self.W2.pre << self.z2.zF
            self.W2.post << self.e1.dmu

            self.W3.pre << self.z3.zF
            self.W3.post << self.e2.dmu
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.q1.batch_size = batch_size
            self.q2.batch_size = batch_size
            self.q3.batch_size = batch_size

            self.Q3.batch_size = batch_size
            self.Q2.batch_size = batch_size
            self.Q1.batch_size = batch_size


            self.Q3.inputs << self.q3.zF
            self.q2.z << self.Q3.outputs
            self.Q2.inputs << self.q2.zF
            self.q1.z << self.Q2.outputs
            self.Q1.inputs << self.q1.zF
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            advance_cmd, advance_args =self.model.compile_by_key(self.q3, self.Q3,
                                                            self.q2, self.Q2,
                                                            self.q1, self.Q1,
                                                            compile_key="advance_state", name="step_advance")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            advance_cmd, advance_args =self.model.compile_by_key(self.E2, self.E1,  ## execute feedback first
                                                               self.z3, self.z2, self.z1,
                                                               self.W3, self.W2, self.W1, ## execute prediction synapses
                                                               self.e2, self.e1, self.e0, ## finally, execute error neurons
                                                               compile_key="advance_state", name='advance_state')

            evolve_cmd, evolve_args =self.model.compile_by_key(self.W1, self.W2, self.W3,
                                                             compile_key="evolve")

            reset_cmd, reset_args =self.model.compile_by_key(self.z3, self.z2, self.z1,
                                                           self.e2, self.e1, self.e0,
                                                           self.W3, self.W2, self.W1,
                                                           self.E1, self.E2,
                                                           self.q1, self.q2, self.q3,
                                                           self.Q1, self.Q2, self.Q3,
                                                           compile_key="reset")

            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.dynamic()

    def dynamic(self):  ## create dynamic commands forself.circuit
        z3, z2, z1, W3, W2, W1, E1, E2, e0, e1, e2, q3, q2, q1, Q3, Q2, Q1 = self.model.get_components("z3", "z2", "z1",
                                                                                 "W3", "W2", "W1",
                                                                                 "E1", "E2",
                                                                                 "e0", "e1", "e2",
                                                                                 "q3", "q2", "q1",
                                                                                 "Q3", "Q2", "Q1")
        self.W1, self.W2, self.W3 = (W1, W2, W3)
        self.e0, self.e1, self.e2 = (e0, e1, e2)
        self.z1, self.z2, self.z3 = (z1, z2, z3)
        self.Q3, self.Q2, self.Q1 = (Q3, Q2, Q1)
        self.q3, self.q2, self.q1 = (q3, q2, q1)
        self.E1, self.E2 = (E1, E2)

        @Context.dynamicCommand
        def clamps(time, X_state):
            self.z3.z.set(time)
            self.q3.z.set(time)
            self.e0.target.set(X_state)


        @Context.dynamicCommand
        def batch_set(batch_size):
            self.z3.batch_size= batch_size
            self.z2.batch_size= batch_size
            self.z1.batch_size = batch_size

            self.e2.batch_size = batch_size
            self.e1.batch_size = batch_size
            self.e0.batch_size = batch_size

            self.W3.batch_size = batch_size
            self.W2.batch_size = batch_size
            self.W1.batch_size = batch_size

            self.E2.batch_size = batch_size
            self.E1.batch_size = batch_size

            self.q1.batch_size = batch_size
            self.q2.batch_size = batch_size
            self.q3.batch_size = batch_size

            self.Q3.batch_size = batch_size
            self.Q2.batch_size = batch_size
            self.Q1.batch_size = batch_size

        self.model.wrap_and_add_command(jit(self.model.evolve), name="evolve")
        # self.model.wrap_and_add_command(jit(self.model.advance_state), name="advance")
        self.model.wrap_and_add_command(jit(self.model.reset), name="reset")
        # self.model.wrap_and_add_command(jit(self.model.step_advance), name="step_advance")


        # @Context.dynamicCommand
        # def l2_grad_wrapper(time, X_state):
        #     self.model.clamps(time, X_state)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), t=3, dt=1.)
        #     return vals[self.e2.mu.path]

        @scanner
        def _process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.model.advance_state(
                compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.W1.outputs.path]


        # @Context.dynamicCommand
        # def clamp_time(time):
        #     self.q3.z.set(time)
            # self.q3.zF.set(time)

        @Context.dynamicCommand
        def jacobian(ts):
            self.q3.z.set(ts)
            vals = self.model.step_advance(Get_Compartment_Batch(), dt=1.)
            vals = self.model.step_advance(vals, t=1, dt=1.)
            # vals = self.model.step_advance(vals, t=1, dt=1.)
            # vals = self.model.step_advance(vals, t=1, dt=1.)
            return vals[self.Q1.outputs.path]


        # @Context.dynamicCommand
        # def jacobian(ts):
        #     self.z3.z.set(ts)
        #     # print('z3.z before training: \n', Get_Compartment_Batch()[self.z3.z.path])
        #     print('z3.zF before training: \n', Get_Compartment_Batch()[self.z3.zF.path].T)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), dt=1.)
        #     # vals = self.model.advance_state(vals, dt=1.)
        #     print('z3.zF after training: \n', vals[self.z3.zF.path].T)
        #     # vals = self.model.advance_state(Get_Compartment_Batch(), dt=1.)
        #
        #     # print('z2.z after training: \n', Get_Compartment_Batch()[self.z2.z.path].shape)
        #
        #     # print(Get_Compartment_Batch()[self.z2.z.path])
        #     input()
        #     # return vals[self.W3.outputs.path]
        #     return vals[self.z3.zF.path]





    # @Context.dynamicCommand
    # def norm(self):
    #     self.W1.weights.set(normalize_matrix(self.W1.weights.value, wnorm=1., order=2, axis=1))
    #     self.W2.weights.set(normalize_matrix(self.W2.weights.value, wnorm=1., order=2, axis=1))
    #     self.W3.weights.set(normalize_matrix(self.W3.weights.value, wnorm=1., order=2, axis=1))

    def process(self, ts, X, ad_dt, scale):

        len_ = len(ts)
        self.model.batch_set(len_)
        self.E1.weights.set(self.W1.weights.value.T)
        self.E2.weights.set(self.W2.weights.value.T)
        self.model.reset()

        self.model.clamps(ts, X)


        # dX = jnp.diag(compute_grad(ts)[0:1, 0, :, 0])[:, None]

        # def grad_fn(ts, z2_, z1_):    # d(W3.outputs) / d(z3.zF) = ts
        #     self.z3.z.set(ts)
        #     # vals2 = self.model.advance_state(Get_Compartment_Batch(), t=3, dt=1.)[self.W3.outputs.path]
        #     self.z2.z.set(z2_)
        #     # vals1 = self.model.advance_state(Get_Compartment_Batch(), t=3, dt=1.)[self.W2.outputs.path]
        #     self.z1.z.set(z1_)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), t=3, dt=1.)
        #     return vals[self.W1.outputs.path]

        # _grad_fn_ = jacrev(grad_fn)

        # def mu1_Z1(inp):    # d(W1.outputs) / d(z1.zF)
        #     self.model.reset()
        #     self.z1.z.set(inp)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), t=3, dt=1.)
        #     return vals[self.W1.outputs.path]


        # def W2_Z2(inp): # d(W2.outputs) / d(z2.zF)
        #     self.model.reset()
        #     self.z2.z.set(inp)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), t=2, dt=1.)
        #     return vals[self.W2.outputs.path]
        #
        # def W3_Z3(ts):    # d(W3.outputs) / d(z3.zF) = ts
        #     # self.model.reset()
        #     self.z3.z.set(ts)
        #     vals = self.model.advance_state(Get_Compartment_Batch(), t=1, dt=1.)
        #     return vals[self.W3.outputs.path]
        #
        # dw1_dz1 = jacrev(W1_Z1)(self.z1.z.value)
        # dw2_dz2 = jacrev(W2_Z2)(self.z2.z.value)
        # dw3_dz3 = jacrev(W3_Z3)(ts)

        z_codes = self.model._process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))
        self.model.evolve(t=self.T, dt=self.dt)

        # self.Q3.weights.set(self.W3.weights.value)
        # self.Q2.weights.set(self.W2.weights.value)
        # self.Q1.weights.set(self.W1.weights.value)
        # self.q2.z.set(self.z2.z.value)
        # self.q1.z.set(self.z1.z.value)
        #
        # print(jnp.diff(ad_dt.ravel())/jnp.diff(ts.ravel()))
        #
        # input()

        # Qmu = self.model.jacobian(ts)

        # print((Qmu - self.W1.outputs.value).T)
        # input()
        # compute_grad = jax.jacrev(self.model.jacobian)
        # dX = jnp.diag(compute_grad(ts)[:, 0, :, 0])[:, None]

        # dX_ = jnp.array(np.gradient(jnp.array(Qmu), jnp.array(ts.ravel()), axis=0))[:, None]
        #
        # print()
        # print()
        # print()
        # print()
        # print()
        # print()
        #
        # print((dX - dX_).T)
        # # print((dX/dX_).T)
        # # print((dX_/dX).T)
        # input()

        # print('q3', self.q3.z.value==self.z3.z.value)
        # print('q2', self.q2.z.value==self.z2.z.value)
        # print('q1', self.q1.z.value==self.z1.z.value)

        # print(self.Q3.weights.value==self.W3.weights.value)
        # print(self.Q2.weights.value == self.W2.weights.value)
        # print(self.Q1.weights.value==self.W1.weights.value)




        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(ts, dnum, label='num', markersize=0.1)
        # ax[0].plot(ts, dX, label='ad')
        # ax[0].plot(ts, dX*(scale*2), label='ad*2')
        # ax[0].plot(ts, dX/(scale*2), label='ad/2')
        # ax[0].legend(loc='best')
        # # ax[1].plot(dnum, dX)
        # plt.show()
        #
        # print(dnum.T)
        # print(dX.T)

        # input()
        dX = 0
        # print(self.W3.dWeights.value.shape,
        #       self.W2.dWeights.value.shape,
        #       self.W1.dWeights.value.shape)
        #
        # input()

        return self.e0.mu.value, dX, self.e0.L.value, self.W3.dWeights.value, self.W2.dWeights.value, self.W1.dWeights.value

    #self.circuit.norm()
    #self.circuit.norm()


    # # # if i % 20 == 0:
    # Xmu = e0.mu.value
    # # # d_Xmu = jnp.array(np.gradient(Xmu, ts_scaled[:, 0], axis=0))# / 2
    # # d_Xmu = jnp.array(np.gradient(Xmu, ts, axis=0))# / 2
    # d_Xmu = dX
    # dict_model.clamp_dx(d_Xmu)
    # dict_model.clamp_output(dictionary_x)
    #
    # dict_model.advance_state(t=T * dt, dt=dt)
    # dict_model.evolve(t=T, dt=dt)
    # dict_model.norm()
