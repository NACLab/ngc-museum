from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess, JointProcess
from ngclearn.components.neurons.graded.rateCell import RateCell
from ngclearn.components.synapses.denseSynapse import DenseSynapse
from ngclearn.components.synapses.hebbian.hebbianSynapse import HebbianSynapse
from ngclearn.components.neurons.graded.gaussianErrorCell import GaussianErrorCell as ErrorCell
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.operations import Summation

from ngcsimlib.global_state import stateManager

class SparseCoding():
    """
    Structure for constructing the sparse coding model proposed in:

    Olshausen, B., Field, D. Emergence of simple-cell receptive field properties
    by learning a sparse code for natural images. Nature 381, 607–609 (1996).

    Note this model imposes a factorial (Cauchy) prior to induce sparsity in the latent
    activities z1 (the latent codebook). Synapses initialized from a (fan-in) scaled
    uniform distribution.
    This model would be named, under the NGC computational framework naming convention
    (Ororbia & Kifer 2022), as the GNCN-t1/SC (SC = sparse coding) or GNCN-t1/Olshausen.

    | Node Name Structure:
    | p(z1) ; z1 -(z1-mu0)-> mu0 ;e0; z0
    | Cauchy prior applied for p(z1)

    Note: You can also recover the model learned through ISTA by using, instead of
    a factorial prior over latents, a thresholding function such as the
    "soft_threshold". (Make sure you set "prior" to "none" in this case.)
    This results in the GNCN-t1/SC emulating a system similar
    to that proposed in:

    Daubechies, Ingrid, Michel Defrise, and Christine De Mol. "An iterative
    thresholding algorithm for linear inverse problems with a sparsity constraint."
    Communications on Pure and Applied Mathematics: A Journal Issued by the
    Courant Institute of Mathematical Sciences 57.11 (2004): 1413-1457.

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        hid_dim: dimensionality of the representation layer of neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics; also
            known as the number of steps to take when conducting iterative
            inference/settling (number of E-steps to take)

        dt: integration time constant

        batch_size: the batch size that the components of this model will operate
            under during simulation calls

        model_type: string indicating what type of sparse-coding model variant
            to configure in this agent constructor; "sc_cauchy" will configure
            the sparse coding model using a Cauchy prior over latent codes, while
            "ista" will configure the iterative-thresholding sprase coding model (ISTA)

        exp_dir: experimental directory to save model results

        load_dir: directory string to load previously trained model from; if not None,
            this will seek a model in the provided directory and build from its
            configuration on disk (default: None)
    """
    # Define Functions
    def __init__(self, dkey, in_dim, hid_dim=100, T=200, dt=1., batch_size=1,
                 model_type="sc_cauchy", exp_dir="exp", load_dir=None, **kwargs):
        model_name = model_type
        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        ## meta-parameters for model dynamics
        self.T = T
        self.dt = dt
        eta_w = 1e-2
        tau_m = 20. # beta = 0.05 = (dt=1)/(tau=20)
        # latent prior type
        act_fx = "identity"
        threshold_type = "none"
        prior_type = "gaussian"
        if model_type == "sc_cauchy":
            ## applies only to Cauchy SC model
            print(" >> Building SC-Cauchy model...")
            prior_type = "cauchy"
            lmbda = 0.14
        else: ## == "ista"
            ## applies only to ISTA model
            print(" >> Building ISTA model...")
            threshold_type = "soft_threshold"
            lmbda = 5e-3

        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:
            with Context("Circuit") as self.circuit:
                self.z1 = RateCell(
                    "z1", n_units=hid_dim, tau_m=tau_m, act_fx=act_fx, prior=(prior_type, lmbda),
                    threshold=(threshold_type, lmbda), integration_type="euler", key=subkeys[0]
                )
                self.e0 = ErrorCell("e0", n_units=in_dim)
                self.W1 = HebbianSynapse(
                    "W1", shape=(hid_dim, in_dim), eta=eta_w, weight_init=DistributionGenerator.fan_in_gaussian(), #dist.fan_in_gaussian(),
                    bias_init=None, w_bound=0., optim_type="sgd", sign_value=-1., key=subkeys[1]
                )
                self.E1 = DenseSynapse(
                    "E1", shape=(in_dim, hid_dim), weight_init=DistributionGenerator.uniform(-0.2, 0.2), #dist.uniform(-0.2, 0.2),
                    resist_scale=1., key=subkeys[2]
                )
                ## since this model will operate with batches, we need to
                ## configure its batch-size here before compiling with the loop-scan
                self.e0.batch_size = batch_size
                self.z1.batch_size = batch_size
                self.W1.batch_size = batch_size
                self.E1.batch_size = batch_size

                ## wire z1.zF to e0.mu via W1
                self.z1.zF >> self.W1.inputs
                self.W1.outputs >> self.e0.mu
                #self.e0.target << self.z0.zF ## no target node exists, so we will clamp instead
                ## wire e0.dmu to z1.j
                self.e0.dmu >> self.E1.inputs
                self.E1.outputs >> self.z1.j
                ## Setup W1 for its 2-factor Hebbian update
                self.z1.zF >> self.W1.pre
                self.e0.dmu >> self.W1.post

                ## Inference process
                advance_process = (MethodProcess(name="advance_process")
                                   >> self.W1.advance_state
                                   >> self.E1.advance_state
                                   >> self.z1.advance_state
                                   >> self.e0.advance_state)
                self.advance = advance_process
                ## Reset-to-baseline process
                reset_process = (MethodProcess(name="reset_process")
                                 >> self.W1.reset
                                 >> self.E1.reset
                                 >> self.z1.reset
                                 >> self.e0.reset)
                self.reset = reset_process
                ## Learning process
                evolve_process = (MethodProcess(name="evolve_process")
                                  >> self.W1.evolve)
                self.evolve = evolve_process

                # processes = (reset_process, advance_process, evolve_process)
                #
                # ## call the compiler to set up jit-i-fied commands and any
                # ## dynamically called command functions
                # self._dynamic(processes)

    def norm(self):
        self.W1.weights.set(normalize_matrix(self.W1.weights.get(), 1., order=2, axis=1))

    def clamp(self, x):
        self.e0.target.set(x)

    # def _dynamic(self, processes): ## create dynamic commands for circuit
    #     W1, E1, e0, z1 = self.circuit.get_components("W1", "E1", "e0", "z1")
    #     self.W1 = W1
    #     self.e0 = e0
    #     self.z1 = z1
    #     self.E1 = E1
    #
    #     reset_proc, advance_proc, evolve_proc = processes
    #
    #     self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
    #     self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")
    #
    #     @Context.dynamicCommand
    #     def clamp(x):
    #         e0.target.set(x)
    #
    #     @Context.dynamicCommand
    #     def norm():
    #         W1.weights.set(normalize_matrix(W1.weights.get(), 1., order=2, axis=1))
    #
    #     @scanner
    #     def process(compartment_values, args):
    #         _t, _dt = args
    #         compartment_values = advance_proc.pure(compartment_values, t=_t, dt=_dt)
    #         return compartment_values, compartment_values[self.z1.zF.path]

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
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            #self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """

        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        # self.advance_proc = self.circuit.get_objects("advance_process", objectType="process")
        processes = self.circuit.get_objects_by_type("process")  ## obtain all saved processes within this context
        self.advance = processes.get("advance_process")
        self.reset = processes.get("reset_process")
        self.evolve = processes.get("evolve_process")

        W1, E1, e0, z1 = self.circuit.get_components("W1", "E1", "e0", "z1")
        self.W1 = W1
        self.E1 = E1
        self.e0 = e0
        self.z1 = z1

        # with Context("Circuit") as self.circuit:
        #     self.circuit.load_from_dir(model_directory)
        #     processes = (self.circuit.reset_process, self.circuit.advance_process, self.circuit.evolve_process)
        #     self._dynamic(processes)

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.get()
        msg = "W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(_W1),
                                                                    jnp.amax(_W1),
                                                                    jnp.mean(_W1),
                                                                    jnp.linalg.norm(_W1))
        return msg

    def viz_receptive_fields(self, fname, field_shape):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            fname: plot fname name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot
        """
        _W1 = self.W1.weights.get().T
        visualize([_W1], [field_shape], self.exp_dir + "/filters/{}".format(fname))

    def process(self, obs, adapt_synapses=True, collect_latent_codes=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T.

        Args:
            obs: observed pattern for SC model to process

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with 2-factor Hebbian plasticity

            collect_latent_codes: if True, will store an T-length array of latent
                code vectors for external analysis

        Returns:
            an array containing spike vectors (will be empty; length = 0 if
                collect_spike_train is False)
        """
        ## check and configure batch size
        ## note: we need to make the components in our model aware of the
        ##       typically dynamic batch size
        self.e0.batch_size = obs.shape[0]
        self.z1.batch_size = obs.shape[0]
        self.W1.batch_size = obs.shape[0]

        ########################################################################
        ## pin/tie feedback synapses to transpose of forward ones
        self.E1.weights.set(jnp.transpose(self.W1.weights.get()))
        self.reset.run() ## reset/set all components to their resting values / initial conditions
        ## Perform several E-steps
        self.clamp(obs) ## clamp data to z0
        inputs = jnp.array(self.advance.pack_rows(self.T, t=lambda x: x, dt=self.dt))
        stateManager.state, outputs = self.advance.scan(inputs)
        ## Perform (optional) M-step (scheduled synaptic updates)
        if adapt_synapses:
            self.evolve.run(t=(self.T + 1) * self.dt, dt=self.dt) ## run synaptic alteration
            self.norm() ## post-update synaptic normalization step
        ########################################################################


        # ########################################################################
        # ## pin/tie feedback synapses to transpose of forward ones
        # self.E1.weights.set(jnp.transpose(self.W1.weights.get()))
        # ## reset/set all components to their resting values / initial conditions
        # self.circuit.reset()
        # ## Perform several E-steps
        # self.circuit.clamp(obs)  ## clamp data to z0
        # z1_codes = self.circuit.process(
        #     jnp.array([[self.dt * i, self.dt] for i in range(self.T)])
        # )
        # ## Perform (optional) M-step (scheduled synaptic updates)
        # if adapt_synapses is True:
        #     self.circuit.evolve(t=self.T, dt=1.)
        #     self.circuit.norm() ## post-update synaptic normalization step
        # ########################################################################


        ## Post-processing / probing desired model outputs
        obs_mu = self.e0.mu.get()  ## get settled prediction
        L0 = self.e0.L.get()  ## calculate prediction loss

        return obs_mu, L0
