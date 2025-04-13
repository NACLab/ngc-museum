from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.utils.model_utils import scanner

from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
from ngcsimlib.compilers import wrap_command, compile_command
from ngclearn.components import (RateCell, HebbianSynapse,
                                 GaussianErrorCell, StaticSynapse)
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.viz.synapse_plot import visualize

class PCRecon():
    """
    Structure for constructing a predictive coding (PC) model for reconstruction tasks.

    Note this model imposes a laplacian prior to induce sparsity in the latent activities
    z1, z2, z3 (the latent codebooks). Synapses are initialized from a (He initialization) gaussian
    distribution with std=sqrt(2/hidden_dim).

    This model would be named, under the NGC computational framework naming convention

    | Node Name Structure:
    | p(z3) ; z3 -(z3-mu2)-> mu2 ;e2; z2
    | p(z2) ; z2 -(z2-mu1)-> mu1 ;e1; z1
    | p(z1) ; z1 -(z1-mu0)-> mu0 ;e0; z0
    | Laplacian prior applied for p(z3), p(z2), p(z1)

    Args:
        dkey: JAX seeding key

        io_dim: input dimensionality

        h1_dim: dimensionality of the 1st representation layer of neuronal cells

        h2_dim: dimensionality of the 2nd representation layer of neuronal cells

        h3_dim: dimensionality of the 3rd (deepest) representation layer of neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics; also
            known as the number of steps to take when conducting iterative
            inference/settling (number of E-steps to take)

        dt: integration time constant

        batch_size: the batch size that the components of this model will operate
            under during simulation calls


        exp_dir: experimental directory to save model results


        load_dir: directory string to load previously trained model from; if not None,
            this will seek a model in the provided directory and build from its
            configuration on disk (default: None)
    """
    # Define Functions
    def __init__(self, dkey, h3_dim, h2_dim, h1_dim, io_dim, T=30, dt=1., batch_size=1,
                 exp_dir="exp", load_dir=None, **kwargs):

        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        self.model_name = "pc_recon"
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        ## meta-parameters for model dynamics
        self.T = T ## number of E-steps to take (stimulus time = T * dt ms)
        self.dt = dt ## integration time constant (ms)
        tau_m = 20.  # beta = 0.05 = (dt=1)/(tau=20) ## time constant for latent trajectories

        self.batch_size = batch_size
        self.eta = 0.005 #1e-2 ## M-step learning rate/step-size

        # latent prior type
        act_fx = "relu"
        opt_type = "sgd"
        prior_type = "laplacian"
        lmbda = 0.14 ## strength of Laplacian prior over latents
        w_bound = 1. ## norm constraint value

        ## He initialization
        w3_init = dist.gaussian(0., jnp.sqrt(2/h3_dim))
        w2_init = dist.gaussian(0., jnp.sqrt(2/h2_dim))
        w1_init = dist.gaussian(0., jnp.sqrt(2/h1_dim))


        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:

            with Context("Circuit") as self.circuit:
                self.z3 = RateCell("z3", n_units=h3_dim, tau_m=tau_m, act_fx=act_fx, prior=(prior_type, lmbda))
                self.W3 = HebbianSynapse("W3", shape=(h3_dim, h2_dim), eta=self.eta, signVal = -1.,
                                    weight_init=w3_init, optim_type=opt_type, w_bound=w_bound, key=subkeys[2])

                self.e2 = GaussianErrorCell("e2", n_units=h2_dim)
                self.z2 = RateCell("z2", n_units=h2_dim, tau_m=tau_m, act_fx=act_fx, prior=(prior_type, lmbda))
                self.W2 = HebbianSynapse("W2", shape=(h2_dim, h1_dim), eta=self.eta, sign_value = -1.,
                                    weight_init=w2_init, optim_type=opt_type, w_bound=w_bound, key=subkeys[0])

                self.e1 = GaussianErrorCell("e1", n_units=h1_dim)
                self.z1 = RateCell("z1", n_units=h1_dim, tau_m=tau_m, act_fx=act_fx, prior=(prior_type, lmbda))
                self.W1 = HebbianSynapse("W1", shape=(h1_dim, io_dim), eta=self.eta, sign_value = -1.,
                                         weight_init=w1_init, optim_type=opt_type, w_bound=w_bound, key=subkeys[1])

                self.e0 = GaussianErrorCell("e0", n_units=io_dim)

                self.E1 = StaticSynapse("E1", shape=(io_dim, h1_dim), key=subkeys[0])
                self.E2 = StaticSynapse("E2", shape=(h1_dim, h2_dim), key=subkeys[1])
                self.E3 = StaticSynapse("E3", shape=(h2_dim, h3_dim), key=subkeys[2])

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                ## since this model will operate with batches, we need to
                ## configure its batch-size here before compiling with the loop-scan
                self.z3.batch_size = batch_size
                self.z2.batch_size = batch_size
                self.z1.batch_size = batch_size

                self.e2.batch_size = batch_size
                self.e1.batch_size = batch_size
                self.e0.batch_size = batch_size

                self.W3.batch_size = batch_size
                self.W2.batch_size = batch_size
                self.W1.batch_size = batch_size

                self.E3.batch_size = batch_size
                self.E2.batch_size = batch_size
                self.E1.batch_size = batch_size

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # wiring

                # f(z3) →🅆3-→ mu2-→ⓔ←-z2
                self.W3.inputs << self.z3.zF
                self.W3.pre << self.z3.zF
                self.e2.mu << self.W3.outputs
                self.e2.target << self.z2.z

                # f(z2) →🅆2-→ mu1-→ⓔ←-z1
                self.W2.inputs << self.z2.zF
                self.W2.pre << self.z2.zF
                self.e1.mu << self.W2.outputs
                self.e1.target << self.z1.z

                # f(z1) →🅆1-→ Xmu-→ⓔ←- X
                self.W1.inputs << self.z1.zF
                self.W1.pre << self.z1.zF
                self.e0.mu << self.W1.outputs

                self.z1.j_td << self.e1.dtarget
                self.W1.post << self.e0.dmu
                self.E1.inputs << self.e0.dmu
                self.z1.j << self.E1.outputs

                self.z2.j_td << self.e2.dtarget
                self.W2.post << self.e1.dmu
                self.E2.inputs << self.e1.dmu
                self.z2.j << self.E2.outputs

                self.W3.post << self.e2.dmu
                self.E3.inputs << self.e2.dmu
                self.z3.j << self.E3.outputs

                reset_process = (JaxProcess(name="reset_process")
                                >> self.z3.reset
                                >> self.z2.reset
                                >> self.z1.reset
                                >> self.e2.reset
                                >> self.e1.reset
                                >> self.e0.reset
                                >> self.W1.reset
                                >> self.W2.reset
                                >> self.W3.reset
                                >> self.E1.reset
                                >> self.E2.reset
                                >> self.E3.reset)
                advance_process = (JaxProcess(name="advance_process")
                                >> self.E1.advance_state
                                >> self.E2.advance_state
                                >> self.E3.advance_state
                                >> self.z3.advance_state
                                >> self.z2.advance_state
                                >> self.z1.advance_state
                                >> self.W3.advance_state
                                >> self.W2.advance_state
                                >> self.W1.advance_state
                                >> self.e2.advance_state
                                >> self.e1.advance_state
                                >> self.e0.advance_state)
                evolve_process = (JaxProcess(name="evolve_process")
                                >> self.W1.evolve
                                >> self.W2.evolve
                                >> self.W3.evolve)

                processes = (reset_process, advance_process, evolve_process)
                self._dynamic(processes)

    def _dynamic(self, processes):  ## create dynamic commands for circuit
        W3, W2, W1, E3, E2, E1, e2, e1, e0, z3, z2, z1 = self.circuit.get_components(
            "W3", "W2", "W1",
            "E3", "E2", "E1",
            "e2", "e1", "e0",
            "z3", "z2", "z1"
        )
        self.W3, self.W2, self.W1 = (W3, W2, W1)
        self.E3, self.E2, self.E1 = (E3, E2, E1)
        self.z3, self.z2, self.z1 = (z3, z2, z1)
        self.e2, self.e1, self.e0 = (e2, e1, e0)

        @Context.dynamicCommand
        def clamp_input(x):
            e0.target.set(x)

        @Context.dynamicCommand
        def norm():
            W1.weights.set(normalize_matrix(W1.weights.value, 1., order=2, axis=1))
            W2.weights.set(normalize_matrix(W2.weights.value, 1., order=2, axis=1))
            W3.weights.set(normalize_matrix(W3.weights.value, 1., order=2, axis=1))

        reset_process, advance_process, evolve_process = processes
        self.circuit.wrap_and_add_command(jit(reset_process.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(evolve_process.pure), name="evolve")
        self.circuit.wrap_and_add_command(jit(advance_process.pure), name="advance")

        @scanner
        def process(compartment_values, args): ## advance is defined within this scan-able process function
            _t, _dt = args
            compartment_values = advance_process.pure(
                compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.z3.zF.path]


    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name, overwrite=True)  ## save current parameter arrays



    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            processes = (self.circuit.reset_process, self.circuit.advance_process, self.circuit.evolve_process)
            self._dynamic(processes)


    def get_synapse_stats(self, W_id='W1'):
        """
        Print basic statistics of the choosed W to string

        Args:
            W_id: the name of the chosen W (synapse); options: "W1" or "W2" or "W3"

        Returns:
            string containing min, max, mean, and L2 norm of Ws
        """
        if W_id == 'W1':
            _W1 = self.W1.weights.value
            msg = "W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W1),
                                                                        jnp.amax(_W1),
                                                                        jnp.mean(_W1),
                                                                        jnp.linalg.norm(_W1))
        if W_id == 'W2':
            _W2 = self.W2.weights.value
            msg = "W2:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W2),
                                                                        jnp.amax(_W2),
                                                                        jnp.mean(_W2),
                                                                        jnp.linalg.norm(_W2))

        if W_id == 'W3':
            _W3 = self.W3.weights.value
            msg = "W3:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W3),
                                                                            jnp.amax(_W3),
                                                                            jnp.mean(_W3),
                                                                            jnp.linalg.norm(_W3))

        return msg



    def viz_receptive_fields(self, fname='receptive_fields'):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            fname: plot file-name name (appended to end of experimental directory)

        """
        _W1 = self.W1.weights.value.T
        visualize([_W1], [(28, 28)], prefix=self.exp_dir + "/filters/{}".format(fname))


    def viz_recons(self, X_test, Xmu_test, fname='recon', field_shape=(28, 28)):
        """
        Generates and saves a plot of the reconstructed images for the
        given test input.

        Args:
            X_test: given input expected to be reconstructed

            Xmu_test: model's reconstruction of the given input

            fname: plot file-name name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot (default=(28, 28))
        """

        visualize([X_test.T, Xmu_test.T], [field_shape, field_shape], prefix=fname)



    def process(self, obs, adapt_synapses=False, collect_latent_codes=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T.

        Args:
            obs: observed pattern for reconstructive PC model to process

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with 2-factor Hebbian plasticity (default=False).

            collect_latent_codes: if True, will store an T-length array of latent
                code of z3 vectors for external analysis (default=False).

        Returns:
            A tuple containing an array of reconstructed signal and a scalar value for reconstructed loss value
            (will be empty; length = 0 if collect_spike_train is False)
        """

        ## pin/tie feedback synapses to transpose of forward ones
        self.E1.weights.set(jnp.transpose(self.W1.weights.value))
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))
        ## reset/set all components to their resting values / initial conditions
        self.circuit.reset()
        ########################################################################
        ## Perform several E-steps
        self.circuit.clamp_input(obs)
        self.z_codes = self.circuit.process(jnp.array([[self.dt * i, self.dt] for i in range(self.T)]))

        ## Perform (optional) M-step (scheduled synaptic updates)
        if adapt_synapses:
            self.circuit.evolve(t=self.T, dt=1.)
            self.circuit.norm()
        ########################################################################
        ## Post-processing / probing desired model outputs
        obs_mu = self.e0.mu.value  ## get reconstructed signal
        L0 = self.e0.L.value  ## calculate reconstruction loss

        if collect_latent_codes:
            return obs_mu, L0, self.z_codes

        return obs_mu, L0
