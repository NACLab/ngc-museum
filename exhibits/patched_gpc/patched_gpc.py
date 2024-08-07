from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize
from jax import random, jit
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import wrap_command
from ngclearn import Context
from ngclearn import numpy as jnp
from ngclearn.utils.model_utils import normalize_matrix
import numpy as np

from ngclearn.components import GaussianErrorCell as ErrorCell
from ngclearn.components import RateCell
from ngclearn.components.synapses import HebbianPatchedSynapse, StaticPatchedSynapse


class HierarchicalPatching_GPC():
    def __init__(self, dkey, D2, D1, D0,
                             n0=1, n1=1, n2=1,
                             weight1_init=None, weight2_init=None, bias_init=None,
                             z2_prior_type=None, z2_lmbda_prior=0., z1_prior_type=None, z1_lmbda_prior=0.,
                             resist_scale1=1., resist_scale2=1., pre_wght1=1., pre_wght2=0.1,
                             w_decay1=0., w_decay2=0., w_bound1=0., w_bound2=0., optim_type1="sgd", optim_type2="sgd",
                             act_fx2="identity", act_fx1="identity", eta1=1e-2, eta2=1e-2, T=200, tau_m1 = 20., tau_m2 = 20., dt=1.,
                             batch_size=1, load_dir=None, exp_dir="exp", **kwargs):

        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        self.T = T
        self.dt = dt

        self.eta1 = eta1
        self.eta2 = eta2
        self.tau_m1 = tau_m1
        self.tau_m2 = tau_m2

        sign_value1, sign_value2 = (-1., -1.)

        self.bias_init = bias_init

        print(">> Building Patching_GPC model with hierarchy ...")

        if load_dir is not None:
            self.load_from_disk(load_dir)
        else:
            with Context("Circuit") as self.circuit:
                # layer 2 neurons
                self.z2 = RateCell("z2", n_units=D2, tau_m=self.tau_m2,
                                           resist_scale=resist_scale2,
                                           prior=((z2_prior_type, z2_lmbda_prior)),
                                           act_fx=act_fx2
                              )
                # layer 1 neurons
                self.e1 = ErrorCell("e1", n_units=D1)
                self.z1 = RateCell("z1", n_units=D1, tau_m=self.tau_m1,
                                           prior=((z1_prior_type, z1_lmbda_prior)),
                                           resist_scale=resist_scale1,
                                           act_fx=act_fx1
                              )
                # input layer
                self.e0 = ErrorCell("e0", n_units=D0)

                # layer 2 synapses
                self.W2 = HebbianPatchedSynapse("W2", shape=(D2, D1), n_sub_models=n2,  
                                                        eta=self.eta2, w_decay=w_decay2,
                                                        pre_wght=pre_wght2, weight_init=weight2_init, bias_init=self.bias_init,
                                                        optim_type=optim_type2, sign_value=sign_value2, w_bound=w_bound2, key=subkeys[3]
                                           )
                self.E2 = StaticPatchedSynapse("E2", shape=(D1, D2), n_sub_models=n2,  
                                                       weight_init=weight2_init, bias_init=self.bias_init, key=subkeys[4])
                # layer 1 synapses
                self.W1 = HebbianPatchedSynapse("W1", shape=(D1, D0), n_sub_models=n1,   
                                                        eta=self.eta1, w_decay=w_decay1,
                                                        pre_wght=pre_wght1, weight_init=weight1_init, bias_init=self.bias_init,
                                                        optim_type=optim_type1, sign_value=sign_value1, w_bound=w_bound1, key=subkeys[5]
                                           )
                self.E1 = StaticPatchedSynapse("E1", shape=(D0, D1), n_sub_models=n1,  
                                                       weight_init=weight1_init, bias_init=self.bias_init, key=subkeys[6])

                self.e0.batch_size = batch_size
                self.e1.batch_size = batch_size
                self.z1.batch_size = batch_size
                self.z2.batch_size = batch_size
                self.W1.batch_size = batch_size
                self.W2.batch_size = batch_size
                self.E1.batch_size = batch_size
                self.E2.batch_size = batch_size

                self.W2.inputs << self.z2.zF
                self.e1.mu << self.W2.outputs
                self.E2.inputs << self.e1.dmu
                self.z2.j << self.E2.outputs

                self.z1.j_td << self.e1.dtarget

                self.W1.inputs << self.z1.zF
                self.e0.mu << self.W1.outputs
                self.E1.inputs << self.e0.dmu
                self.z1.j << self.E1.outputs

                self.e1.target << self.z1.z

                self.W1.pre << self.z1.zF
                self.W1.post << self.e0.dmu
                self.W2.pre << self.z2.zF
                self.W2.post << self.e1.dmu

                advance_cmd, advance_args = self.circuit.compile_by_key( self.E1, self.E2, ## execute feedback first
                                                                                     self.z2, self.z1, ## execute state neurons
                                                                                     self.W2, self.W1, ## execute prediction synapses
                                                                                    self.e1, self.e0, ## finally, execute error neurons
                                                                           compile_key="advance_state")

                evolve_cmd, evolve_args = self.circuit.compile_by_key(self.W1, self.W2,
                                                                      compile_key="evolve")

                reset_cmd, reset_args = self.circuit.compile_by_key(self.z1, self.z2, self.e0, self.e1,
                                                                                self.W1, self.W2, self.E1, self.E2,
                                                                    compile_key="reset")

                self.dynamic()


    def dynamic(self):  ## create dynamic commands for circuit
        W1, W2, E1, E2, e0, e1, z1, z2 = self.circuit.get_components("W1", "W2",
                                                                                     "E1", "E2",
                                                                                     "e0", "e1",
                                                                                     "z1", "z2")
        self.W1, self.W2 = (W1, W2)
        self.e0, self.e1 = (e0, e1)
        self.z1, self.z2 = (z1, z2)
        self.E1, self.E2 = (E1, E2)

        @Context.dynamicCommand
        def clamp(x):
            e0.target.set(x)

        @Context.dynamicCommand
        def norm():
            W1.weights.set(normalize_matrix(W1.weights.value, 1., order=2, axis=1))
            W2.weights.set(normalize_matrix(W2.weights.value, 1., order=2, axis=1))

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")

        @scanner
        def process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)

            return compartment_values, compartment_values[self.z1.zF.path]


    def save_to_disk(self, params_only=False):
        if params_only is True:
            model_dir = "{}/{}/custom".format(self.exp_dir, "HGPC")
            self.W1.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, "HGPC")  ## save current parameter arrays


    def load_from_disk(self, model_directory):
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory)
            self.dynamic()


    def get_synapse_stats(self, W_id='W1'):
        if W_id == 'W1':
            _W1 = self.W1.weights.value
            msg = "W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W1),
                                                                        jnp.amax(_W1),
                                                                        jnp.mean(_W1),
                                                                        jnp.linalg.norm(_W1))
            return msg

        if W_id == 'W2':
            _W2 = self.W2.weights.value
            msg = "W2:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W2),
                                                                        jnp.amax(_W2),
                                                                        jnp.mean(_W2),
                                                                        jnp.linalg.norm(_W2))
            return msg

    def viz_receptive_fields(self, rf1_shape):

        d2_, d1_, d0_ = (self.W2.sub_shape[0], self.W1.sub_shape[0], self.W1.sub_shape[1])
        d2, d1, d0 = d2_, d1_, d0_

        w1 = self.W1.weights.value
        w2 = self.W2.weights.value

        w_padding = jnp.zeros((d1_, 80))
        w1_pad = np.concatenate([np.hstack([w_padding,
                                            w1[i * d1:(i + 1) * d1, i * d0:(i + 1) * d0],
                                            w_padding]) for i in range(self.W1.n_sub_models)])

        RF1 = np.hstack([w1.T[i * d0: (i + 1) * d0,
                              i * d1: (i + 1) * d1] for i in range(self.W1.n_sub_models)])

        RF2 = w1_pad.T @ w2.T
        visualize([RF2, RF1], sizes=[(16, 26), (rf1_shape)], order=['F', 'C'], prefix='rf2_rf1')


    def process(self, sensory_in, adapt_synapses=True):

        self.e0.batch_size = sensory_in.shape[0]
        self.e1.batch_size = sensory_in.shape[0]
        self.z1.batch_size = sensory_in.shape[0]
        self.z2.batch_size = sensory_in.shape[0]
        self.W1.batch_size = sensory_in.shape[0]
        self.W2.batch_size = sensory_in.shape[0]
        self.E1.batch_size = sensory_in.shape[0]
        self.E2.batch_size = sensory_in.shape[0]

        self.E1.weights.set(jnp.transpose(self.W1.weights.value))
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))

        self.circuit.reset()
        # ########################################################################
        self.circuit.clamp(sensory_in)
        z1_codes = self.circuit.process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))

        if adapt_synapses is True:
            self.circuit.evolve(t=self.T, dt=1.)
            self.circuit.norm()

        predicted_mu = self.e0.mu.value

        return predicted_mu, (self.e0.L.value, self.e1.L.value)
