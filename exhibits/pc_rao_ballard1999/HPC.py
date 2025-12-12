import os
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import (RateCell, HebbianPatchedSynapse as BackwardSynapse,
                                 GaussianErrorCell, StaticPatchedSynapse as ForwardSynapse)
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.viz.synapse_plot import visualize

from ngcsimlib.global_state import stateManager
from ngclearn import MethodProcess, JointProcess, Context
from ngclearn.utils.distribution_generator import DistributionGenerator as dist


class HierarchicalPredictiveCoding():
    """
    Structure for constructing a predictive coding (PC) model for reconstruction tasks.

    Note this model imposes a laplacian prior to induce sparsity in the latent activities
    z1, z2 (the latent codebooks). Synapses are initialized from a (He initialization) gaussian
    distribution with std=sqrt(2/hidden_dim).

    This model would be named, under the NGC computational framework naming convention

    | Node Name Structure:
    | p(z2) ; z2 -(z2-mu1)-> mu1 ;e1; z1
    | p(z1) ; z1 -(z1-mu0)-> mu0 ;e0; z0
    | prior type applied for p(z2), p(z1)

    Args:
        dkey: JAX seeding key

        h2_dim: full dimensionality of the 2nd representation layer of neuronal cells

        h1_dim: full dimensionality of the 1st representation (lowest) layer of neuronal cells

        in_dim: input dimensionality

        n_p2: number of dense synaptic modules in 2nd layer of neuronal cells

        n_p1: number of dense synaptic modules in 1st (lowest) layer of neuronal cells

        n_inPatch: number of patches in input image

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
    def __init__(self, dkey,
                 h2_dim, h1_dim, in_dim,
                 n_p2=1, n_p1=1, n_inPatch=1,
                 T=30, dt=1.,
                 batch_size=1,
                 sigma=1.,
                 sigma_td=1.,
                 k2=0.05,                       ## M-step learning rate/step-size
                 k1 = 0.05,                     ## time constant for latent trajectories
                 act_fx="identity",             ## neural activation function
                 r1_prior=("laplacian", 0.),
                 r2_prior=("laplacian", 0.),
                 U_prior = ("gaussian", 0.),
                 model_name = "PC_Model",
                 exp_dir="exp",
                 load_dir=None,
                 reset_exp_dir=True,
                 **kwargs):

        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        if reset_exp_dir: # remove everything and create experiment directory from scratch
            makedir(exp_dir)
            makedir(exp_dir + "/filters")
            makedir(exp_dir + "/img_recons")
            makedir(exp_dir + "/raster")
            print(" > Created experiment directory at ", exp_dir)
        else:
            print(" > Using existing experiment directory at ", exp_dir, " or creating if non-existent.")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(exp_dir + "/filters", exist_ok=True)
            os.makedirs(exp_dir + "/img_recons", exist_ok=True)
            os.makedirs(exp_dir + "/raster", exist_ok=True)

        self.model_name = model_name

        ############ meta-parameters for model structure and dynamics
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

        self.n_inPatch = n_inPatch
        self.n_p1 = n_p1
        self.n_p2 = n_p2

        self.inPatch_dim = in_dim // n_inPatch
        self.batch_size = batch_size

        self.T = T                         ## number of E-steps to take (stimulus time = T * dt ms)
        self.dt = dt                       ## neural activity integration time constant (ms)
        #############   initialization and design choices
        tau_m = 1. / k1
        w_bound = 1.                                            ## norm constraint value
        opt_type = "sgd"                                        ##   synaptic (weights) optimization type

        w2_init = dist.gaussian(mean=0., std=jnp.sqrt(2/h2_dim))         ## He initialization for layer-2 synapses
        w1_init = dist.gaussian(mean=0., std=jnp.sqrt(2/h1_dim))         ## He initialization for layer-1  synapses

        #################################################################
        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:
            with Context("Circuit") as self.circuit:
                self.z2 = RateCell("z2", n_units=h2_dim, tau_m=tau_m, act_fx=act_fx, prior=r2_prior)
                self.z1 = RateCell("z1", n_units=h1_dim, tau_m=tau_m, act_fx=act_fx, prior=r1_prior)

                self.W2 = BackwardSynapse("W2",
                                         shape=(h2_dim, h1_dim),
                                         n_sub_models=self.n_p2,     ## number of modules in the layer
                                         sign_value=-1.,                  ## -1 means M-step solve minimization problem
                                         optim_type=opt_type,
                                         eta=k2,
                                         weight_init=w2_init,
                                         prior = U_prior,
                                         w_bound=w_bound,
                                         key=subkeys[1]
                                         )
                self.W1 = BackwardSynapse("W1",
                                         shape=(h1_dim, in_dim),
                                         n_sub_models=n_p1,              ## number of modules in the layer
                                         sign_value=-1.,                 ## -1 means M-step solve minimization problem
                                         optim_type=opt_type,
                                         eta=k2,
                                         weight_init=w1_init,
                                         prior = U_prior,
                                         w_bound=w_bound,
                                         key=subkeys[0]
                                         )

                self.e0 = GaussianErrorCell("e0", n_units=in_dim, sigma = sigma)
                self.e1 = GaussianErrorCell("e1", n_units=h1_dim, sigma = sigma_td)

                self.E2 = ForwardSynapse("E2", shape=(h1_dim, h2_dim), n_sub_models=self.n_p2, key=subkeys[1])
                self.E1 = ForwardSynapse("E1", shape=(in_dim, h1_dim), n_sub_models=self.n_p1, key=subkeys[0])

                ############################################################
                ## since this model will operate with batches, we need to
                ## its batch-size here before compiling with the loop-scan
                self.batch_setup()

                ############################################################
                # wiring (signal pathway is according to Rao & Ballard 1999 paper)
                ############################################################
                ######### feedback (Top-down) #########
                self.z2.zF >> self.W2.inputs
                self.z1.zF >> self.W1.inputs

                ## Top-down prediction
                self.W2.outputs >> self.e1.mu
                self.W1.outputs >> self.e0.mu
                ## actual neural activation
                self.z1.z >> self.e1.target

                ## Top-down prediction errors
                self.e1.dtarget >> self.z1.j_td

                ######### forward (Bottom-up) #########
                ## feedforward the errors via synapses
                self.e1.dmu >> self.E2.inputs
                self.e0.dmu >> self.E1.inputs
                ## Bottom-up modulated errors
                self.E2.outputs >> self.z2.j
                self.E1.outputs >> self.z1.j

                ######## Hebbian learning #########
                ## Pre Synaptic Activation
                self.z2.zF >> self.W2.pre
                self.z1.zF >> self.W1.pre
                ## Post Synaptic residual error
                self.e1.dmu >> self.W2.post
                self.e0.dmu >> self.W1.post

                ############################################################
                self.reset_process = (MethodProcess(name="reset_process")
                                    >> self.z2.reset
                                    >> self.z1.reset
                                    >> self.e1.reset
                                    >> self.e0.reset
                                    >> self.W1.reset
                                    >> self.W2.reset
                                    >> self.E1.reset
                                    >> self.E2.reset
                                    )
                self.advance_process = (MethodProcess(name="advance_process")
                                    >> self.E1.advance_state
                                    >> self.E2.advance_state
                                    >> self.z2.advance_state
                                    >> self.z1.advance_state
                                    >> self.W2.advance_state
                                    >> self.W1.advance_state
                                    >> self.e1.advance_state
                                    >> self.e0.advance_state
                                    )
                self.evolve_process = (MethodProcess(name="evolve_process")
                                    >> self.W1.evolve
                                    >> self.W2.evolve
                                    )


    def batch_setup(self):
        batch_size = self.batch_size

        self.z2.batch_size = batch_size
        self.z1.batch_size = batch_size

        self.e1.batch_size = batch_size
        self.e0.batch_size = batch_size

        self.W2.batch_size = batch_size
        self.W1.batch_size = batch_size

        self.E2.batch_size = batch_size
        self.E1.batch_size = batch_size

    def clamp_input(self, x):
      self.e0.target.set(x)

    def norm(self):
      self.W1.weights.set(normalize_matrix(self.W1.weights.get(), 1., order=2, axis=1))
      self.W2.weights.set(normalize_matrix(self.W2.weights.get(), 1., order=2, axis=1))

    def _advance_process(self, obs):
      self.clamp_input(obs)
      inputs = jnp.array(self.advance_process.pack_rows(self.T, t=lambda x: x, dt=self.dt))
      stateManager.state, z_codes = self.advance_process.scan(inputs)
      return z_codes

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)


    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        with self.circuit:
            ## obtain all saved processes within this context
            processes = self.circuit.get_objects_by_type("process")
            self.advance_process = processes.get("advance_process")
            self.reset_process = processes.get("reset_process")
            self.evolve_process = processes.get("evolve_process")

            W2, W1, E2, E1, e1, e0, z2, z1 = self.circuit.get_components("W2", "W1",
                                                                         "E2", "E1",
                                                                         "e1", "e0",
                                                                         "z2", "z1"
                                                                         )
            self.W2, self.W1 = (W2, W1)
            self.E2, self.E1 = (E2, E1)
            self.z2, self.z1 = (z2, z1)
            self.e1, self.e0 = (e1, e0)

            self.batch_setup()

    def get_synapse_stats(self, W_id='W1'):
        """
        Print basic statistics of the choosed W to string

        Args:
            W_id: the name of the chosen W (synapse); options: "W1" or "W2"

        Returns:
            string containing min, max, mean, and L2 norm of Ws
        """
        if W_id == 'W1':
            _W1 = self.W1.weights.get()
            msg = "W1: ---Sparsity {} \n  min {} ;  max {} \n  mu {} ;  norm {}".format(
                                                    100 * (jnp.sum(jnp.where(_W1 == 0, 1, 0)) // _W1.size),
                                                    jnp.amin(_W1),
                                                    jnp.amax(_W1),
                                                    jnp.mean(_W1),
                                                    jnp.linalg.norm(_W1))
        if W_id == 'W2':
            _W2 = self.W2.weights.get()
            msg = "W2: ---Sparsity {} \n  min {} ;  max {} \n  mu {} ;  norm {}".format(
                                                    100 * (jnp.sum(jnp.where(_W2 == 0, 1, 0)) // _W2.size),
                                                    jnp.amin(_W2),
                                                    jnp.amax(_W2),
                                                    jnp.mean(_W2),
                                                    jnp.linalg.norm(_W2))
        return msg


    def viz_receptive_fields(self, patch_shape, stride_shape=(0, 0), max_filter=100, order="C", fname='receptive_fields'):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            patch_shape: input image patch shape

            stride_shape: the overlapping dimensions of input image patches

            max_filter: maximum number of receptive fields to visualize

            fname: plot file-name name (appended to end of experimental directory)

        """


        _W1 = self.W1.weights.get()

        if self.n_p1 == 1:
            visualize([_W1.T], [patch_shape], prefix=self.exp_dir + "/filters/{}".format(fname))

        else:
            d1_pre, d1_erf = self.W1.sub_shape
            list_filter = []
            for i in range(self.n_p1):
                rf_pre = _W1[
                            i * d1_pre:(i+1) * d1_pre,
                            i * d1_erf:(i+1) * d1_erf
                            ]
                list_filter.append(rf_pre)
            h1_rf = jnp.array(list_filter)     # (n_modules, dim_module, (n_inPatch * dim_inPatch))
            rf_vis = h1_rf.reshape(-1, self.inPatch_dim)

            visualize([rf_vis[:max_filter, :].T], [patch_shape], order=order, prefix=self.exp_dir + "/filters/{}".format(fname))



    def viz_recons(self, X_test, Xmu_test, image_shape=(28, 28), fname='recon'):
        """
        Generates and saves a plot of the reconstructed images for the
        given test input.

        Args:
            X_test: given input expected to be reconstructed

            Xmu_test: model's reconstruction of the given input

            patch_shape: the shape of cropped image (patches)

            fname: plot file-name name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot (default=(28, 28))
        """

        X_test = X_test.reshape(-1, image_shape[0] * image_shape[1])
        Xmu_test = Xmu_test.reshape(-1, image_shape[0] * image_shape[1])

        visualize([X_test.T, Xmu_test.T], [image_shape, image_shape], prefix=self.exp_dir + "/img_recons/{}".format(fname))


    def process(self, obs, adapt_synapses=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T.

        Args:
            obs: observed pattern for reconstructive PC model to process

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with 2-factor Hebbian plasticity (default=False).

            collect_latent_codes: if True, will store an T-length array of latent
                code of z2 vectors for external analysis (default=False).

        Returns:
            A tuple containing an array of reconstructed signal and a scalar value for reconstructed loss value
            (will be empty; length = 0 if collect_spike_train is False)
        """


        # ═══════════════════════════════════════════════════════════════════════════
        # WEIGHT TYING: Pin feedback synapses to transpose of forward weights (E = W.T)
        # ═══════════════════════════════════════════════════════════════════════════
        self.E1.weights.set(jnp.transpose(self.W1.weights.get()))
        self.E2.weights.set(jnp.transpose(self.W2.weights.get()))

        # ═══════════════════════════════════════════════════════════════════════════
        # INITIALIZATION: Reset/set all components to resting state  / initial conditions
        # ═══════════════════════════════════════════════════════════════════════════
        self.reset_process.run()

        # ───────────────────────────────────────────────────────────────────────────
        # INFERENCE PHASE: Perform several E-steps to compute latent codes
        # ───────────────────────────────────────────────────────────────────────────
        self.z_codes = self._advance_process(obs)

        # ───────────────────────────────────────────────────────────────────────────
        # LEARING PHASE (optional): Perform one M-step (scheduled synaptic updates)
        # ───────────────────────────────────────────────────────────────────────────
        if adapt_synapses:
            self.evolve_process.run(t=self.T, dt=self.dt)
            self.norm()      ## normalize synapses so each row's l2-norm =1

        # ═══════════════════════════════════════════════════════════════════════════
        # POST-PROCESSING: probing desired model outputs
        # ═══════════════════════════════════════════════════════════════════════════
        obs_mu = self.e0.mu.get()               ## get reconstructed signal
        L0 = self.e0.L.get()                    ## get reconstruction loss


        return obs_mu, L0
