from jax import random, jit
import numpy as np
from ngclearn.utils.io_utils import makedir

from ngclearn.utils import weight_distribution as dist
from ngclearn import Context, numpy as jnp
from ngclearn.components import (RateCell,
                                 HebbianSynapse,
                                 GaussianErrorCell,
                                 StaticSynapse)
from ngclearn.utils.model_utils import scanner
from ngclearn.modules.regression.lasso import Iterative_Lasso as Lasso
from ngclearn.modules.regression.elastic_net import Iterative_ElasticNet as ElasticNet
from ngclearn.modules.regression.ridge import Iterative_Ridge as Ridge
from ngcsimlib.compilers.process import Process



class DeepMoD():
    """
        Structure for constructing the Deep learning driven Model Discovery:
        Both, Gert-Jan, Gijs Vermarien, and Remy Kusters. "Sparsely constrained
        neural networks for model discovery of PDEs." arXiv preprint
        arXiv:2011.04336 (2020).


        Note this model decouples the network constraint of the differential
        equation terms and the sparsity selection process, allowing for more flexible and
        robust model discovery by first calculating a sparsity mask and then constraining
        the network only with active terms.

        (The original paper was Deep learning driven Model Discovery (DeepMoD):
        Both, Gert-Jan, et al. "DeepMoD: Deep learning for model discovery
        in noisy data." Journal of Computational Physics 428 (2021): 109985.)


        | Node Name Structure:
        | z3 -(W3)-> e2, z2 -(W2)-> e1, z1 -(W1)-> e0;
        | e2 -(E2)-> z2 <- e1, e1 -(E1)-> z1 <- e0
        | Note: W1, W2, W3 -> Hebbian-adapted synapses


    Args:
        dkey: JAX seeding key

        ts: Time series data points

        dict_dim: Dimensionality of the dictionary/library space

        lib_creator: Library creator function for creating candidate functions out of the predicted values (Xmu)

        in_dim: Input dimensionality

        h1_dim: Dimensionality of first hidden layer

        h2_dim: Dimensionality of second hidden layer

        out_dim: Output dimensionality

        batch_size: Number of samples to process in each batch

        w_fill: Initial weight fill value (Default: 0.05)

        lr: Learning rate for optimization (Default: 0.01)

        lmbda: Regularization parameter (Default: 0.0001)

        l1_ratio: Elastic net mixing parameter (Default: 0.0)

        optim_type: Type of optimizer to use (Default: "adam")

        threshold: Threshold for sparse coefficient selection (Default: 0.001)

        scale: Scaling factor for dictionary terms (Default: 2.0)

        solver_name: Type of regression solver ("lasso", "elastic_net", or "ridge") (Default: "lasso")

        eta: Learning rate for Hebbian updates (Default: 1e-3)

        tau_m: Membrane time constant (Default: 20.0)

        T: Number of discrete time steps for simulation (Default: 50)

        dt: Integration time step (Default: 1.0)

        exp_dir: Directory path for saving experimental results (Default: "exp")

        model_name: Name identifier for the model (Default: "deepmod")

        """
    def __init__(self, key, ts, dict_dim, lib_creator, in_dim, h1_dim, h2_dim, out_dim, batch_size,
                 w_fill=0.05, lr=0.01, lmbda=0.0001, l1_ratio=0., optim_type="adam", threshold=0.001, scale=2.,
                 solver_name = "lasso", eta = 1e-3, tau_m = 20., T=50, dt=1.,
                 model_name="deepmod", **kwargs):
        dkey, *subkeys = random.split(key, 10)

        self.model_name = model_name
        self.solver_name = solver_name
        self.nodes = None
        self.threshold = threshold

        ## meta-parameters for model dynamics
        self.T = T
        self.dt = dt
        self.ts = ts
        self.eta = eta
        self.lib_creator = lib_creator

        if solver_name == "lasso" or solver_name == "l1":
            print(" >> Building Lasso solver model...")
            self.scale = scale
            epochs = 100
            sys_dim = out_dim
            self.method_params = (key, self.solver_name, sys_dim, dict_dim, batch_size, w_fill, lr,
                                  lmbda, optim_type, threshold, epochs)

            self.solver = Lasso(*self.method_params)
            self.W_init = self.solver.W.weights.value


        if solver_name == "elastic_net" or solver_name == "l1l2":
            print(" >> Building Elastic-Net solver model...")

            self.scale = scale
            epochs = 100
            sys_dim = out_dim
            self.method_params = (key, self.solver_name, sys_dim, dict_dim, batch_size, w_fill, lr,
                                  lmbda, l1_ratio, optim_type, threshold, epochs)

            self.solver = ElasticNet(*self.method_params)


        if solver_name == "ridge" or solver_name == "l2":
            print(" >> Building Ridge solver model...")
            self.scale = scale
            epochs = 100
            sys_dim = out_dim
            self.method_params = (key, self.solver_name, sys_dim, dict_dim, batch_size, w_fill, lr,
                                  lmbda, optim_type, threshold, epochs)

            self.solver = Ridge(*self.method_params)

        opt_type = "adam"
        act_fx = "sine"
        self.omega_0 = 30  # check 2-300-10

        W3_dist = dist.uniform(
                                amin=-1 / h2_dim,
                                amax=1 / h2_dim
                               )
        W2_dist = dist.uniform(
                                amin=-np.sqrt(6 / h1_dim) / self.omega_0,
                                amax=np.sqrt(6 / h1_dim) / self.omega_0
                               )
        W1_dist = dist.uniform(
                                amin=-np.sqrt(6 / out_dim) / self.omega_0,
                                amax=np.sqrt(6 / out_dim) / self.omega_0
                               )

        with Context(self.model_name) as self.model:
            ############ L3
            self.z3 = RateCell("z3", n_units=in_dim, tau_m=tau_m , act_fx="identity")
            self.W3 = HebbianSynapse("W3", shape=(in_dim, h2_dim), eta=eta, w_bound=0., signVal=-1, sign_value=-1,
                                     optim_type=opt_type, weight_init=W3_dist, key=subkeys[0]
                                     )
            ############ L2
            self.e2 = GaussianErrorCell("e2", n_units=h2_dim)
            self.z2 = RateCell("z2", n_units=h2_dim, tau_m=tau_m , act_fx=act_fx, omega_0=self.omega_0,
                                                                                        batch_size=batch_size)

            self.W2 = HebbianSynapse("W2", shape=(h2_dim, h1_dim), eta=eta, w_bound=0., signVal=-1, sign_value=-1,
                                     optim_type=opt_type, weight_init=W2_dist, key=subkeys[1])
            self.E2 = StaticSynapse("E2", shape=(h1_dim, h2_dim)
                                    )
            ############ L1
            self.e1 = GaussianErrorCell("e1", n_units=h1_dim)
            self.z1 = RateCell("z1", n_units=h1_dim, tau_m=tau_m , act_fx="identity")
            self.W1 = HebbianSynapse("W1", shape=(h1_dim, out_dim), eta=eta, w_bound=0., signVal=-1, sign_value=-1,
                                     optim_type=opt_type, weight_init=W1_dist, key=subkeys[2])
            self.E1 = StaticSynapse("E1", shape=(out_dim, h1_dim)
                                    )
            ############ input
            self.e0 = GaussianErrorCell("e0", n_units=out_dim
                                        )
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Replace compile_by_key with Process definitions
            advance_process = (Process(name="advance_process")
                              >> self.E2.advance_state  # execute feedback first
                              >> self.E1.advance_state
                              >> self.z3.advance_state
                              >> self.z2.advance_state
                              >> self.z1.advance_state
                              >> self.W3.advance_state  # execute prediction synapses
                              >> self.W2.advance_state
                              >> self.W1.advance_state
                              >> self.e2.advance_state  # finally, execute error neurons
                              >> self.e1.advance_state
                              >> self.e0.advance_state)

            evolve_process = (Process(name="evolve_process")
                             >> self.W1.evolve
                             >> self.W2.evolve
                             >> self.W3.evolve)

            reset_process = (Process(name="reset_process")
                            >> self.z3.reset
                            >> self.z2.reset
                            >> self.z1.reset
                            >> self.e2.reset
                            >> self.e1.reset
                            >> self.e0.reset
                            >> self.W3.reset
                            >> self.W2.reset
                            >> self.W1.reset
                            >> self.E1.reset
                            >> self.E2.reset)

            # Store processes for use in dynamic method
            processes = (reset_process, advance_process, evolve_process)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self._dynamic(processes)

    def _dynamic(self, processes):  ## create dynamic commands for the model
        z3, z2, z1, W3, W2, W1, E1, E2, e0, e1, e2 = self.model.get_components("z3", "z2", "z1",
                                                                                 "W3", "W2", "W1",
                                                                                 "E1", "E2",
                                                                                 "e0", "e1", "e2")
        self.W1, self.W2, self.W3 = (W1, W2, W3)
        self.e0, self.e1, self.e2 = (e0, e1, e2)
        self.z1, self.z2, self.z3 = (z1, z2, z3)
        self.E1, self.E2 = (E1, E2)

        reset_proc, advance_proc, evolve_proc = processes

        @Context.dynamicCommand
        def clamps(input, target):
            self.z3.z.set(input)
            self.e0.target.set(target)

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

        # Replace the old command wrappers with the new process-based ones
        self.model.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")
        self.model.wrap_and_add_command(jit(advance_proc.pure), name="advance_state")
        self.model.wrap_and_add_command(jit(reset_proc.pure), name="reset")

        @scanner
        def _process(compartment_values, args):
            _t, _dt = args
            compartment_values = advance_proc.pure(compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.W1.outputs.path]


    def prediction_process(self, input, target):
        self.model.batch_set(len(input))
        self.E1.weights.set(self.W1.weights.value.T)
        self.E2.weights.set(self.W2.weights.value.T)

        self.model.reset()
        self.model.clamps(input, target)

        self.model._process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))
        self.model.evolve(t=self.T, dt=self.dt)

        return self.e0.mu.value, self.e0.L.value


    def thresholding(self):
        coef_old = self.solver.W.weights.value
        coef_new = jnp.where(jnp.abs(coef_old) >= self.threshold, coef_old, 0.)

        self.solver.W.weights.set(coef_new)

        return coef_new


    def process(self, ts_scaled, X):
        self.model.batch_set(len(ts_scaled))
        Xmu, loss = self.prediction_process(input=self.ts, target=X)

        library, _ = self.lib_creator.fit([Xmu[:, i] for i in range(Xmu.shape[1])])
        dXmu = jnp.array(np.gradient(jnp.array(Xmu), self.ts.ravel(), axis=0))

        coef = self.solver.fit(y=dXmu/self.scale, X=library)[0]

        return coef, loss
