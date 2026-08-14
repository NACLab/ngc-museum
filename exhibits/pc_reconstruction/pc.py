import os
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
import numpy as np
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.viz.synapse_plot import visualize
from ngcsimlib.global_state import stateManager
from ngclearn import MethodProcess, Context
from ngclearn.components import (RateCell, HebbianPatchedSynapse, GaussianErrorCell)
from ngclearn.components.input_encoders.ganglionCell import _reconstruct as reconstruct
from ngclearn.components.input_encoders.ganglionCell import RetinalGanglionCell

from ngclearn.utils.analysis.effective_dim import rankme
from ngclearn.utils.analysis import LinearProbe
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, plot_latents


class HierarchicalPredictiveCoding():
    """
    Structure for constructing a predictive coding (PC) model for reconstruction tasks.

    Note this model imposes a laplacian prior to induce sparsity in the latent activities
    z1, z2, z3 (the latent codebooks). Synapses are initialized from a (He initialization) gaussian
    distribution with std=sqrt(2/hidden_dim).

    This model would be named, under the NGC computational framework naming convention

    | Node Name Structure:
    | p(z3) ; z3 -(z3-mu2)-> mu2 ;e2; z2
    | p(z2) ; z2 -(z2-mu1)-> mu1 ;e1; z1
    | p(z1) ; z1 -(z1-mu0)-> mu0 ;e0
    | prior type applied for p(z3), p(z2), p(z1)

    Args:
        dkey: JAX seeding key

        h3_dim: full dimensionality of the 3rd (deepest) representation layer of neuronal cells

        h2_dim: full dimensionality of the 2nd representation layer of neuronal cells

        h1_dim: full dimensionality of the 1st representation (lowest) layer of neuronal cells

        in_dim: input dimensionality

        n_p3: number of dense synaptic modules in 3rd (deepest) layer of neuronal cells

        n_p2: number of dense synaptic modules in 2nd layer of neuronal cells

        n_p1: number of dense synaptic modules in 1st (lowest) layer of neuronal cells

        n_inPatch: number of patches in input image

        area_shape: receptive field area of ganglion cells in this module all together

        patch_shape: each ganglion cell receptive field area

        step_shape: the non-overlapping area between each two ganglion cells

        input_encoder: string name of input preprocessing kernel (Default: identity)

        input_encoder_sigma: standard deviation of gaussian kernel

        T: number of discrete time steps to simulate neuronal dynamics; also
            known as the number of steps to take when conducting iterative
            inference/settling (number of E-steps to take)

        dt: integration time constant

        tau_m: membrane/state time constant (milliseconds)

        lr: or eta, global learning rate for synaptic update

        sigma_e2: variance of the prediction in layer-2 (assumes isotropic multivariate gaussian distribution,
                  then in this layer, covariance matrix (𝚺) collapses to a sigma * I)

        sigma_e1: variance of the prediction in layer-1 (assumes isotropic multivariate gaussian distribution,
                  then in this layer, covariance matrix (𝚺) collapses to a sigma * I)

        sigma_e0: variance of the prediction in layer-0 (assumes isotropic multivariate gaussian distribution,
                  then in this layer, covariance matrix (𝚺) collapses to a sigma * I)

        act_fx: string name of activation function/nonlinearity to use

        r3_prior: a kernel for specifying the type of distribution to impose over
                  layer-3 neuronal dynamics (Default: ("laplacian", 0.))

        r2_prior: a kernel for specifying the type of distribution to impose over
                  layer-2 neuronal dynamics (Default: ("laplacian", 0.))

        r1_prior: a kernel for specifying the type of distribution to impose over
                  layer-1 neuronal dynamics (Default: ("laplacian", 0.))


        synaptic_prior: prior applied to all synapses for reqularization during update/learning.
        (default: ("gaussian", 0.))

        circuit_name: string indicating the name of the pc circuit type

        batch_size: the batch size that the components of this model will operate
            under during simulation calls

        exp_dir: experimental directory to save model results


        load_dir: directory string to load previously trained model from; if not None,
            this will seek a model in the provided directory and build from its
            configuration on disk (default: None)
    """
    # Define Functions
    def __init__(self, dkey,
                 # ═══════════  Architecture parameters ════════════
                 h3_dim=1, h2_dim=1, h1_dim=1, in_dim=1,
                 n_p3=1, n_p2=1, n_p1=1, n_inPatch=1,
                 batch_size=1,
                 # ═══════════════ Preprocessing parameters ═════════════
                 area_shape=(28, 28),
                 patch_shape=(28, 28),
                 step_shape=(0, 0),
                 input_encoder = None,
                 input_encoder_sigma = 0.,
                 # ════════════ Neurons/Cells parameters ════════════
                 T=30, dt=1.,
                 tau_m=20,                    ## time constant for latent trajectories
                 act_fx="identity",           ## neural activation function
                 sigma_e2=1., sigma_e1=1., sigma_e0=1.,
                 r3_prior=(None, 0.),
                 r2_prior=(None, 0.),
                 r1_prior=(None, 0.),
                 # -------------   Lateral parameters  -------------
                 use_lateral=False,
                 adaptive_lateral=False,
                 exc_inh=(0., 0.),
                 lat_eta=0.,
                 # ═══════════   Synaptses parameters  ════════════
                 lr=0.05,                    ## M-step learning rate/step-size
                 synaptic_prior=("gaussian", 0.),
                 w_opt_type="sgd",
                 # ═════════   Experimental parameters ════════════
                 circuit_name = "Circuit",
                 exp_dir="exp",
                 load_dir=None,
                 reset_exp_dir=True,
                 **kwargs):

        ## ═════════════════════════ Simiulation Initializations ════════════════════════
        dkey, *subkeys = random.split(dkey, 10)
        self.circuit_name = circuit_name
        self.exp_dir = exp_dir

        ## ═══════════════════════  Model reading/writing directories  ═══════════════════
        if reset_exp_dir: # remove everything and create experiment directory from scratch
            makedir(exp_dir)
            makedir(exp_dir + "/filters")
            makedir(exp_dir + "/img_recons")
            makedir(exp_dir + "/analysis")
            print(" > Created experiment directory at ", exp_dir)
        else:
            print(" > Using existing experiment directory at ", exp_dir, " or creating if non-existent.")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(exp_dir + "/filters", exist_ok=True)
            os.makedirs(exp_dir + "/img_recons", exist_ok=True)
            os.makedirs(exp_dir + "/analysis", exist_ok=True)

        ## ══════════════ meta-parameters for model structure ══════════════
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.h3_dim = h3_dim

        self.n_inPatch = n_inPatch
        self.n_p1 = n_p1
        self.n_p2 = n_p2
        self.n_p3 = n_p3

        self.area_shape = area_shape
        self.patch_shape = patch_shape
        self.step_shape = step_shape
        self.batch_size = batch_size
        self.inPatch_dim = in_dim // n_inPatch
        ## ═══════════════ meta-parameters for model dynamic ═════════════
        self.input_encoder = input_encoder
        self.gauss_sigma = input_encoder_sigma

        self.T = T                         ## number of E-steps to take (stimulus time = T * dt ms)
        self.dt = dt                       ## neural activity integration time constant (ms)

        exc, inh = exc_inh
        d3, d2, d1 = h3_dim // n_p3, h2_dim // n_p2, h1_dim // n_p1
        exc_inh3 = (exc / (d3 - 1), inh / (d3 - 1)) if d3 > 1 else 0   ## block_dim(z3) == d3
        exc_inh2 = (exc / (d2 - 1), inh / (d2 - 1)) if d2 > 1 else 0   ## block_dim(z2) == d2
        exc_inh1 = (exc / (d1 - 1), inh / (d1 - 1)) if d1 > 1 else 0   ## block_dim(z1) == d1

        ## ═════════════════════ Synaptses parameters ═══════════════════
        w_bound = 0.                                                     ## norm constraint value

        # ═════════════════════════════════════════════════════════
        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:
            with Context(self.circuit_name) as self.circuit:
                self.RGC = RetinalGanglionCell("RGC", filter_type=self.input_encoder,
                                               area_shape = self.area_shape,
                                               n_cells=self.n_inPatch,
                                               patch_shape=self.patch_shape,
                                               step_shape=self.step_shape,
                                               sigma = self.gauss_sigma,
                                               batch_size=batch_size
                                               )

                # ════════════════════════════════════════════
                self.e0 = GaussianErrorCell("e0", n_units=in_dim, sigma=sigma_e0, batch_size=batch_size)

                self.z1 = RateCell("z1", n_units=h1_dim, batch_size=batch_size,
                                   tau_m=tau_m, act_fx=act_fx, prior=r1_prior,
                                   use_lateral=use_lateral,
                                   adaptive_lateral=adaptive_lateral,
                                   n_patch=self.n_p1,
                                   exc_inh=exc_inh1,
                                   Wl_eta=lat_eta
                                   )
                self.e1 = GaussianErrorCell("e1", n_units=h1_dim, sigma=sigma_e1, batch_size=batch_size)

                self.z2 = RateCell("z2", n_units=h2_dim, batch_size=batch_size,
                                   tau_m=tau_m, act_fx=act_fx, prior=r2_prior,
                                   use_lateral=use_lateral,
                                   adaptive_lateral=adaptive_lateral,
                                   n_patch=self.n_p2,
                                   exc_inh=exc_inh2,
                                   Wl_eta=lat_eta
                                   )
                self.e2 = GaussianErrorCell("e2", n_units=h2_dim, sigma=sigma_e2, batch_size=batch_size)

                self.z3 = RateCell("z3", n_units=h3_dim, batch_size=batch_size,
                                   tau_m=tau_m, act_fx=act_fx, prior=r3_prior,
                                   use_lateral=use_lateral,
                                   adaptive_lateral=adaptive_lateral,
                                   n_patch=self.n_p3,
                                   exc_inh=exc_inh3,
                                   Wl_eta=lat_eta
                                   )

                # ════════════════════════════════════════════
                self.W3 = HebbianPatchedSynapse("W3", shape=(h3_dim, h2_dim), batch_size=batch_size,
                                                 n_sub_models=self.n_p3,      ## number of modules in the layer
                                                 prior=synaptic_prior,
                                                 optim_type=w_opt_type,
                                                 w_bound=w_bound,
                                                 sign_value=-1.,              ## -1 means M-step solve minimization problem
                                                 eta=lr,
                                                 key=subkeys[2]
                                                 )
                self.W2 = HebbianPatchedSynapse("W2", shape=(h2_dim, h1_dim), batch_size=batch_size,
                                                n_sub_models=self.n_p2,      ## number of modules in the layer
                                                prior=synaptic_prior,
                                                optim_type=w_opt_type,
                                                w_bound=w_bound,
                                                sign_value=-1.,              ## -1 means M-step solve minimization problem
                                                eta=lr,
                                                key=subkeys[1]
                                                )
                self.W1 = HebbianPatchedSynapse("W1", shape=(h1_dim, in_dim), batch_size=batch_size,
                                                 n_sub_models=n_p1,            ## number of modules in the layer
                                                 prior=synaptic_prior,
                                                 optim_type=w_opt_type,
                                                 w_bound=w_bound,
                                                 sign_value=-1.,               ## -1 means M-step solve minimization problem
                                                 eta=lr,
                                                 key=subkeys[0]
                                                 )

                # ═════════════════════════════════════════════════════════
                ## since this model will operate with batches, we need to
                ## its batch-size here before compiling with the loop-scan
                self.batch_setup()

                # ═════════════════════════════════════════════════════════
                # wiring (signal pathway is according to Rao & Ballard 1999 paper)
                # ══════ feedback (Top-down) ═════════════
                self.z3.zF >> self.W3.inputs
                self.z2.zF >> self.W2.inputs
                self.z1.zF >> self.W1.inputs

                # ══════ Top-down prediction ═════════════
                if self.h3_dim == 1:
                    self.z2.z >> self.e2.mu
                else:
                    self.W3.outputs >> self.e2.mu
                self.W2.outputs >> self.e1.mu
                self.W1.outputs >> self.e0.mu

                # ══════ actual neural activation ════════
                self.z2.z >> self.e2.target
                self.z1.z >> self.e1.target
                self.RGC.outputs >> self.e0.target

                # ══════ Top-down prediction errors ══════
                self.e1.dtarget >> self.z1.j_td
                self.e2.dtarget >> self.z2.j_td

                # ══════ forward (Bottom-up) ═════════════
                ## feedforward the errors via synapses
                self.e2.dmu >> self.W3.post_in
                self.e1.dmu >> self.W2.post_in
                self.e0.dmu >> self.W1.post_in

                # ══════ Bottom-up modulated errors ══════
                self.W3.pre_out >> self.z3.j
                self.W2.pre_out >> self.z2.j
                self.W1.pre_out >> self.z1.j

                # ═════════ Hebbian learning ═════════════
                # ══════ Pre Synaptic Activation ══════
                self.z3.zF >> self.W3.pre
                self.z2.zF >> self.W2.pre
                self.z1.zF >> self.W1.pre
                # ══════ Post Synaptic residuals ══════
                self.e2.dmu >> self.W3.post
                self.e1.dmu >> self.W2.post
                self.e0.dmu >> self.W1.post

                ## ═════════════════════════════════════════════════════════════════
                self.reset_process = (MethodProcess(name="reset_process")
                                >> self.RGC.reset
                                >> self.z3.reset
                                >> self.z2.reset
                                >> self.z1.reset
                                >> self.e2.reset
                                >> self.e1.reset
                                >> self.e0.reset
                                >> self.W1.reset
                                >> self.W2.reset
                                >> self.W3.reset
                                )
                self.advance_process = (MethodProcess(name="advance_process")
                                >> self.RGC.advance_state
                                >> self.z3.advance_state
                                >> self.z2.advance_state
                                >> self.z1.advance_state
                                >> self.W3.advance_state
                                >> self.W2.advance_state
                                >> self.W1.advance_state
                                >> self.e2.advance_state
                                >> self.e1.advance_state
                                >> self.e0.advance_state
                                )
                self.evolve_process = (MethodProcess(name="evolve_process")
                                >> self.W1.evolve
                                >> self.W2.evolve
                                >> self.W3.evolve
                                >> self.z1.evolve
                                >> self.z2.evolve
                                >> self.z3.evolve
                                )

    def batch_setup(self):
        batch_size = self.batch_size
        self.RGC.batch_size = batch_size

        self.z3.batch_size = batch_size
        self.z2.batch_size = batch_size
        self.z1.batch_size = batch_size

        self.e2.batch_size = batch_size
        self.e1.batch_size = batch_size
        self.e0.batch_size = batch_size

        self.W3.batch_size = batch_size
        self.W2.batch_size = batch_size
        self.W1.batch_size = batch_size

        self.RGC.reset()
        self.z3.reset()
        self.z2.reset()
        self.z1.reset()
        self.e2.reset()
        self.e1.reset()
        self.e0.reset()
        self.W1.reset()
        self.W2.reset()
        self.W3.reset()

        # print(f"batch setup: {self.RGC.inputs.get().shape}")

    def clamp_stimuli(self, x):
        # print(f"clamp_stimuli: x shape: {x.shape}, RGC input shape: {self.RGC.inputs.get().shape}")
        self.RGC.inputs.set(x)

    def norm(self):
      self.W1.weights.set(normalize_matrix(self.W1.weights.get(), 1., order=2, axis=1))
      self.W2.weights.set(normalize_matrix(self.W2.weights.get(), 1., order=2, axis=1))
      self.W3.weights.set(normalize_matrix(self.W3.weights.get(), 1., order=2, axis=1))

    def _advance_process(self, obs):
      # several E-steps, can use for loop or scan
      self.clamp_stimuli(obs)
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
            model_dir = "{}/{}/component/custom".format(self.exp_dir, "model")
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name="model", overwrite=True)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit = Context.load(directory=model_directory, module_name="model")
        with self.circuit:
            processes = self.circuit.get_objects_by_type("process") ## obtain all saved processes within this context
            self.advance_process = processes.get("advance_process")
            self.reset_process = processes.get("reset_process")
            self.evolve_process = processes.get("evolve_process")
            RGC, W3, W2, W1, e2, e1, e0, z3, z2, z1 = self.circuit.get_components(
                "RGC", "W3", "W2", "W1",
                "e2", "e1", "e0",
                "z3", "z2", "z1"
            )

            self.RGC = RGC
            self.W3, self.W2, self.W1 = (W3, W2, W1)
            self.z3, self.z2, self.z1 = (z3, z2, z1)
            self.e2, self.e1, self.e0 = (e2, e1, e0)
            self.batch_setup()
            # print(f"after loading: bs: {self.batch_size}. rgc bs: {self.RGC.outputs.get().shape}")

    def get_synapse_stats(self):
        """
        Print basic statistics of the choosed W to string

        Args:
            W_id: the name of the chosen W (synapse); options: "W1" or "W2" or "W3"

        Returns:
            string containing min, max, mean, std, L2 norm, and non-zero elements of synapses
        """
        header = f"  {'W':<4} {'[#patches]':<4} {'non-Zero%':>10} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'L2 norm':>12}"
        sep = "-" * len(header)
        _W = [(self.W1.name, self.W1.shape[0], self.W1.n_sub_models, self.W1.weights.get()),
             (self.W2.name, self.W2.shape[0], self.W2.n_sub_models, self.W2.weights.get()),
             (self.W3.name, self.W3.shape[0], self.W3.n_sub_models, self.W3.weights.get())]
        print("\n\n" + sep)
        print(header)
        print(sep)
        for W_name, W_dim, n_ptch, W in _W:
            if W_dim > 1:
                n_nonzero = float(jnp.sum(jnp.where(W == 0, 0, 1)))
                size = float(W.size)
                nonzero_pct = 100.0 * n_nonzero / size
                w_min = float(jnp.min(W))
                w_max = float(jnp.max(W))
                w_mean = float(jnp.mean(W))
                w_std = float(jnp.std(W))
                w_norm = float(jnp.linalg.norm(W))
                print(f"  {W_name:<4}   {n_ptch:<5} {nonzero_pct:>9.2f}% {w_min:>10.4f} {w_max:>10.4f} "
                      f"{w_mean:>10.4f} {w_std:>10.4f} {w_norm:>12.4f}")
        print(sep + "\n\n")

    def viz_receptive_fields(self, vis_effective_RF=False, vis_brain_in=False, max_n_vis=-1, fname='receptive_fields'):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            patch_shape: input image patch shape

            step_shape: the overlapping dimensions of input image patches

            max_filter: maximum number of receptive fields to visualize

            fname: plot file-name name (appended to end of experimental directory)

        """
        _W1 = self.W1.weights.get()

        d2, d1, d0 = (self.W2.sub_shape[0], self.W1.sub_shape[0], self.W1.sub_shape[1])
        n2, n1, n0 = (self.W3.n_sub_models, self.W2.n_sub_models, self.W1.n_sub_models)

        ix, iy = self.area_shape
        px, py = self.patch_shape
        sx, sy = self.step_shape

        n_row = px + (n0 - 1) * sx              ## 16 = 16 + (2 * 0)
        n_col = py + (n0 - 1) * sy              ## 26 = 16 + (2 * 5)
        area_shape = (n_row, n_col)             ## (16 , 26)

        if sx > 0:
            pad_x = jnp.zeros((sx, n_col))
        if sy > 0:
            pad_y = jnp.zeros((n_row, sy))

        ## ═════════════════════════════════════════════════════════════════
        if n0 == 1:
            ## >>>>>>>>>>>>>>>>>>>>>>>>>>>
            visualize([_W1[:max_n_vis].T], [self.patch_shape], prefix=self.exp_dir + "/filters/{}".format(fname))

        ## ═════════════════════════════════════════════════════════════════
        else:
            list_filter = []
            for i in range(n0):
                rf_pre = _W1[i * d1:(i+1) * d1, i * d0:(i+1) * d0]
                list_filter.append(rf_pre)
            h1_rf = jnp.array(list_filter)

            RF1 = jnp.hstack([_W1[:max_n_vis].T[i * d0: (i + 1) * d0,
                              i * d1: (i + 1) * d1] for i in range(self.W1.n_sub_models)]
                             )[:, :max_n_vis]

            ## >>>>>>>>>>>>>>>>>>>>>>>>>>
            visualize([RF1],
                      sizes=[(self.patch_shape)],
                      order=['C'],
                      prefix=self.exp_dir + "/filters/{}".format(fname))

            ## ══════════════════════════════════════════════════════════════
            if vis_effective_RF:
                erf_list = []
                for n in range(h1_rf.shape[1]):
                    erf_v2 = h1_rf[:, n, :]
                    A = erf_v2[0].reshape(*self.patch_shape)
                    B = erf_v2[1].reshape(*self.patch_shape)
                    C = erf_v2[2].reshape(*self.patch_shape)

                    Apad = jnp.concatenate([A, pad_y, pad_y], axis=1)
                    Bpad = jnp.concatenate([pad_y, B, pad_y], axis=1)
                    Cpad = jnp.concatenate([pad_y, pad_y, C], axis=1)

                    effective_rf = Apad + Bpad + Cpad
                    erf_list.append(effective_rf)
                erf_array = jnp.array(erf_list).reshape(-1, area_shape[0]*area_shape[1])[:max_n_vis]

                ## >>>>>>>>>>>>>>>>>>>>>>>>>>
                visualize([erf_array.T],
                          sizes=[(area_shape)],
                          order=['C'],
                          prefix=self.exp_dir + "/filters/L2_{}".format(fname))

                ## ══════════════════════════════════════════════════════════════
                if vis_brain_in:
                    rgc_out = self.RGC.outputs.get().reshape(-1, self.RGC.n_cells, *self.patch_shape)

                    A = rgc_out[:, 0, :, :].reshape(-1, *self.patch_shape)
                    B = rgc_out[:, 1, :, :].reshape(-1, *self.patch_shape)
                    C = rgc_out[:, 2, :, :].reshape(-1, *self.patch_shape)

                    pad_iy = jnp.zeros((A.shape[0], n_row, sy))

                    Apad = jnp.concatenate([A, pad_iy, pad_iy], axis=2)
                    Bpad = jnp.concatenate([pad_iy, B, pad_iy], axis=2)
                    Cpad = jnp.concatenate([pad_iy, pad_iy, C], axis=2)

                    I_in = Apad + Bpad + Cpad
                    brain_in = I_in.reshape(-1, ix * iy)[:max_n_vis, :]

                    obs = self.RGC.inputs.get().reshape(-1, ix * iy)[:max_n_vis, :]
                    ## >>>>>>>>>>>>>>>>>>>>>>>>>>
                    visualize([obs.T, brain_in.T],
                              sizes=[(area_shape), (area_shape)],
                              order=['C', 'C'],
                              prefix=self.exp_dir + "/filters/L2_{}".format(fname))
                    ## ══════════════════════════════════════════════════════════════

    def viz_recons(self, X_test, Xmu_test, image_shape=(28, 28),  image_area_step=(0, 0), fname='recon'):
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

        ix, iy = image_shape
        ax, ay = self.area_shape
        px, py = self.patch_shape
        sx, sy = self.step_shape

        s_ax, s_ay = image_area_step

        ## ══════════════════════════════════════════════════════════════
        ## if each area is patched
        if self.area_shape != self.patch_shape:
            nx = ax // px if sx == 0 else (ax - px) // sx + 1
            ny = ay // py if sy == 0 else (ay - py) // sy + 1

            Xmu_test = Xmu_test.reshape(-1, nx * ny, px, py)
            Xmu_test = reconstruct(Xmu_test, (nx, ny), area_shape=self.area_shape,
                                                       patch_shape=self.patch_shape,
                                                       step_shape=self.step_shape)

        ## ══════════════════════════════════════════════════════════════
        ## if numbers of areas croped per image is > 1
        if image_shape != self.area_shape:
            n_ax = ix // ax if s_ax==0 else (ix - ax) // s_ax + 1
            n_ay = iy // ay if s_ay==0 else (iy - ay) // s_ay + 1

            X_test = X_test.reshape(-1, n_ax * n_ay, ax, ay)
            X_test = reconstruct(X_test, (n_ax, n_ay), area_shape=image_shape,
                                                       patch_shape=self.area_shape,
                                                       step_shape=(s_ax, s_ay))

            Xmu_test = Xmu_test.reshape(-1, n_ax * n_ay, ax, ay)
            Xmu_test = reconstruct(Xmu_test, (n_ax, n_ay), area_shape=image_shape,
                                                           patch_shape=self.area_shape,
                                                           step_shape=(s_ax, s_ay))

        ## ══════════════════════════════════════════════════════════════
        ## areas are retrieved
        X_test = X_test.reshape(-1, ix * iy)
        Xmu_test = Xmu_test.reshape(-1, ix * iy)

        visualize([X_test.T, Xmu_test.T], [image_shape, image_shape], prefix=self.exp_dir + "/img_recons/{}".format(fname))

    def get_latents(self):
        return {"z1": self.z1.z.get(),
                "z2": self.z2.z.get(),
                "z3": self.z3.z.get()}


    def process_test(self, X):
        Zs = []
        Lb = 0
        mb_size = self.batch_size
        for p in range(0, (X.shape[0] // mb_size) * mb_size, mb_size):
            self.process(X[p:p + mb_size], adapt_synapses=False)
            Lb += self.e0.L.get()
            Zs.append(self.get_latents())
        loss = Lb / len(X)
        return Zs, loss

    def collect_latents(self, Z, Y, save=True, fname="latents_codes"):
        Y = Y[:len(Z) * self.batch_size]
        Z = {name: jnp.concatenate([z[name] for z in Z], axis=0) for name in Z[0]}
        if save:
            self.save_codes(Z, Y, fname=fname)
        return Z, Y

    def save_codes(self, Z, Y, fname="latents_codes"):
        self.latents_path = f"{self.exp_dir}/analysis"
        np.savez(f"{self.latents_path}/{fname}.npz", Y=np.asarray(Y),
                 **{k: np.asarray(v) for k, v in Z.items()})

    def load_codes(self, fname="latents_codes"):
        codes = np.load(f"{self.latents_path}/{fname}.npz")
        Z = {n: codes[n] for n in codes.files if n != "Y"}
        Y = jnp.asarray(codes["Y"])
        return Z, Y


    def get_eff_dims(self, fname="latents_codes"):
        Z = self.load_codes(fname)[0]
        eff_dims = {}
        print("\n-------- Effective Dimensionality --------")
        for name, z in Z.items():
            eff_dim = float(rankme(z))
            eff_dims[name] = float(rankme(z))
            print(f"{name!r}: {eff_dim:>8.2f}     (% {100*eff_dim/z.shape[1]:>5.2f})")
        print("------------------------------------------ \n")
        return eff_dims


    def get_probe_acc(self, fname="latents_codes"):
        Z, Y = self.load_codes(fname)
        dkey = random.PRNGKey(42)
        C = Y.shape[1]
        n_train = int(0.8 * Y.shape[0]) // self.batch_size * self.batch_size
        n_test = (Y.shape[0] - n_train) // self.batch_size * self.batch_size

        print("\n===============  Latents Probe Accuracy =================")
        for name, z in Z.items():
            Z_train, Y_train, Z_test, Y_test = z[:n_train], Y[:n_train], z[n_train:n_train + n_test], Y[n_train:n_train + n_test]
            args = dict(source_seq_length=1, input_dim=z.shape[1], out_dim=C, batch_size=self.batch_size, dev_batch_size=self.batch_size)

            print(f"{name!r} Linear Probe")
            LinearProbe(dkey, use_softmax=False, **args).fit((Z_train, Y_train), (Z_test, Y_test), n_iter=20)
            print(f"{name!r} Sofmax Probe")
            LinearProbe(dkey, use_softmax=True, **args).fit((Z_train, Y_train), (Z_test, Y_test), n_iter=20)
            print()
        print("==============================================================")


    def plot_codes(self, fname="latents_codes"):
        Z, Y = self.load_codes(fname)
        if fname == "latents_init":
            for name, z in Z.items():
                plot_latents(extract_tsne_latents(np.asarray(z), verbose=0), np.asarray(Y),
                             plot_fname=f"{self.latents_path}/tsne_{name}_init.jpg", alpha=0.3, cmap='tab10')
        else:
            for name, z in Z.items():
                plot_latents(extract_tsne_latents(np.asarray(z), verbose=0), np.asarray(Y),
                             plot_fname=f"{self.latents_path}/tsne_{name}.jpg", alpha=0.3, cmap='tab10')



    def process(self, obs, adapt_synapses=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T.

        Args:
            obs: observed pattern for reconstructive PC model to process

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with 2-factor Hebbian plasticity (default=False).

            collect_latent_codes: if True, will store an T-length array of latent
                code of the most top z vectors for external analysis (default=False).

        Returns:
            A tuple containing an array of reconstructed signal and a scalar value for reconstructed loss value
            (will be empty; length = 0 if collect_spike_train is False)
        """

        # ══════  Inference  ════════════════════════════════════════
        #####     Reset
        self.reset_process.run()

        ## Perform several E-steps
        self._advance_process(obs)

        # ══════  Learning  ════════════════════════════════════════
        #####     Perform M-step - (optional)
        if adapt_synapses:
            self.evolve_process.run(t=self.T, dt=self.dt)
            self.norm()

        # ═══════════════════════════════════════════════════════════
        obs_mu = self.e0.mu.get()   ## get reconstructed signal

        return obs_mu











