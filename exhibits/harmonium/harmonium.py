from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess, JointProcess
from custom.bernoulliStochasticCell import BernoulliStochasticCell
from ngclearn.components.synapses.denseSynapse import DenseSynapse
from ngclearn.components.synapses.hebbian.hebbianSynapse import HebbianSynapse

from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
#from ngcsimlib.global_state import stateManager


@jit
def get_param_updates( ## contrastive Hebbian learning recipe
        weights, dWpos, dWneg, post_pos, post_neg, pre_pos, pre_neg, l1_lambda, l2_lambda
):
    N = post_pos.shape[0] * 1.
    dW = (dWpos - dWneg) * (1./N) ## contrast pos & neg Hebbian updates
    l1_Wdecay = -jnp.sign(weights) * l1_lambda ## sparse weight decay term
    l2_Wdecay = -weights * l2_lambda * 0.5
    dWeights = (dW + l2_Wdecay + l1_Wdecay)
    dhidbias = jnp.sum(post_pos - post_neg, axis=0, keepdims=True) * (1./N)
    dvisbias = jnp.sum(pre_pos - pre_neg, axis=0, keepdims=True) * (1./N)
    return -dWeights, -dhidbias, -dvisbias

class Harmonium():
    """
        Structure for constructing the harmonium (restricted Boltzmann machine; RBM) model proposed in:

        | Hinton, Geoffrey E. "Training products of experts by maximizing contrastive likelihood." Technical Report,
        | Gatsby computational neuroscience unit (1999).

        | Node Name Structure:
        | z1 -(z1-z0)-> z0
        | z0 -(z0-z1)-> z1
        | Note: z1-z0 = (z0-z1)^T (transpose-tied synapses)

        Another important reference for designing stable Harmoniums is here:

        | Hinton, Geoffrey E. "A practical guide to training restricted Boltzmann machines." Neural networks: Tricks of
        | the trade. Springer, Berlin, Heidelberg, 2012. 599-619.

        Note that this model exhibit specifically implements contrastive divergence (CD-1) for adapting the synapses
        of this model during training.

        Args:
            dkey: JAX seeding key
            obs_dim: number of latent variables in layer z0 (or sensory x)
            hid_dim: number of latent variables in layer z1
            eta: learning rate to control strength of applied CD/CHL updates
            l1_lambda: L1 regularization strength applied to synaptic updates (Default: 0)
            l2_lambda: L2 regularization strength applied to synaptic updates (Default: 0)
            is_meanfield: is this RBM/harmonium to be treated as a mean-field model? (Default: False)
            exp_dir: experimental directory to save model results
            load_dir: directory to load model from, overrides initialization/model object creation if
                non-None (Default: None)
        """

    def __init__(
            self, dkey, obs_dim=1, hid_dim=100, eta=0.01, l1_lambda=0., l2_lambda=0., is_meanfield=False, exp_dir="exp",
            load_dir=None, **kwargs
    ):
        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        self.model_name = "rbm"
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        self.is_meanfield = is_meanfield
        self.eta = eta
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        sigma = 0.01 # 0.02
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:
            ## A harmonium can be set up as two contrasting "graphs" or co-models:
            ## the 1st is the positive-phase, data-dependent co-model
            ## the 2nd is the negative-phase, data-independent co-model
            ## and CHL is essentially subtracting out the data-independent (co-)model 
            ## from the data-dependent one
            with Context("Circuit") as self.circuit:
                ## set up positive-phase graph
                self.z0 = BernoulliStochasticCell("z0", n_units=obs_dim, is_stoch=False)
                self.z1 = BernoulliStochasticCell("z1", n_units=hid_dim, key=subkeys[0])

                self.W1 = HebbianSynapse(
                    "W1", shape=(obs_dim, hid_dim), eta=0., weight_init=dist.gaussian(mean=0., std=sigma),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type="sgd", sign_value=1., key=subkeys[1]
                )
                self.W1.biases.set(self.W1.biases.get() - 4.) ## to crudely encourage sparsity
                self.z0.s >> self.W1.inputs
                self.W1.outputs >> self.z1.inputs

                ## set up negative-phase graph
                self.z0neg = BernoulliStochasticCell("z0neg", n_units=obs_dim, key=subkeys[3])
                self.z1neg = BernoulliStochasticCell("z1neg", n_units=hid_dim, key=subkeys[4])

                self.E1 = DenseSynapse( ## E1 = W1.T
                    "E1", shape=(hid_dim, obs_dim), weight_init=dist.gaussian(mean=0., std=sigma),
                    bias_init=dist.constant(value=0.), resist_scale=1., key=subkeys[2]
                )
                self.E1.weights.set(self.W1.weights.get().T)
                self.V1 = HebbianSynapse( ## V1 = W1
                    "V1", shape=(obs_dim, hid_dim), eta=0., weight_init=dist.gaussian(mean=0., std=sigma),
                    bias_init=None, w_bound=0., optim_type="sgd", sign_value=1., key=subkeys[1]
                )
                self.V1.weights.set(self.W1.weights.get())
                self.V1.biases.set(self.W1.biases.get())

                if is_meanfield:
                    self.z1.p >> self.E1.inputs
                else:
                    self.z1.s >> self.E1.inputs
                self.E1.outputs >> self.z0neg.inputs
                if is_meanfield:
                    self.z0neg.p >> self.V1.inputs
                else:
                    self.z0neg.p >> self.V1.inputs ## drive hiddens by probs of visibles
                    #self.z0neg.s >> self.V1.inputs
                self.V1.outputs >> self.z1neg.inputs

                ## positive-phase inference
                self.advance_pos = (MethodProcess(name="advance_pos")
                                    >> self.z0.advance_state
                                    >> self.W1.advance_state
                                    >> self.z1.advance_state)
                ## negative-phase inference
                self.advance_neg = (MethodProcess(name="advance_neg")
                                    >> self.E1.advance_state
                                    >> self.z0neg.advance_state
                                    >> self.V1.advance_state
                                    >> self.z1neg.advance_state)

                ## set up contrastive Hebbian learning rule (pos-stats - neg-stats)
                self.z0.s >> self.W1.pre
                self.z1.p >> self.W1.post
                self.z0neg.p >> self.V1.pre
                self.z1neg.p >> self.V1.post

                # self.z0.s >> self.W1.pre
                # self.z1.s >> self.W1.post
                # self.z0neg.s >> self.V1.pre
                # self.z1neg.s >> self.V1.post

                self.calc_update = (MethodProcess(name="calc_update")
                                    >> self.W1.calc_update
                                    >> self.V1.calc_update)

                ## set up reset function
                self.reset = (MethodProcess(name="reset")
                              >> self.z0.reset
                              >> self.z1.reset
                              >> self.z0neg.reset
                              >> self.z1neg.reset
                              >> self.W1.reset
                              >> self.E1.reset
                              >> self.V1.reset)

            optim_type = "adam"
            self.opt_params = get_opt_init_fn(optim_type)([self.W1.weights.get(), self.W1.biases.get(), self.E1.biases.get()])
            self.opt = get_opt_step_fn(optim_type, eta=self.eta)

    def init_vis_biases(self, X):
        """
        Initialize this RBM's visible biases in accordance with a data design matrix X. Specifically, this computes,
        for each visible bias (v_i) value:

        | v_i = log(p_i / (1 - p_i)), where p_i is the mean probability value of feature (column) i in dataset X

        Args:
            X: design matrix to estimate visible biases over
        """
        p_v = jnp.mean(X, axis=0, keepdims=True)
        m_v = (p_v > 0.0)
        p_v = p_v * m_v + (1. - m_v) * 0.01
        log_p_v = jnp.log(p_v / (1. - p_v))
        self.E1.biases.set(log_p_v)  ## init visible biases to feature proportions

    def clamp_input(self, x):
        """
        Clamps input pattern(s) x to this model.

        Args:
            x: input patterns to clamp
        """
        self.z0.inputs.set(x)

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only: ## this condition allows to only write actual parameter values w/in components to disk
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.E1.save(model_dir)
            self.V1.save(model_dir)
        else: ## this saves the whole model form (JSON structure as well as parameter values)
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        #self.advance_proc = self.circuit.get_objects("advance_process", objectType="process")
        processes = self.circuit.get_objects_by_type("process") ## obtain all saved processes within this context
        self.advance_pos = processes.get("advance_pos")
        self.advance_neg = processes.get("advance_neg")
        self.reset = processes.get("reset")
        self.calc_update = processes.get("calc_update")

        W1, E1, V1, z0, z1, z0neg, z1neg = self.circuit.get_components(
            "W1", "E1", "V1", "z0", "z1", "z0neg", "z1neg"
        )
        self.W1 = W1
        self.E1 = E1
        #self.E1.weights.set(self.W1.weights.get().T)
        self.V1 = V1
        #self.V1.weights.set(self.W1.weights.get())
        self.z0 = z0
        self.z1 = z1
        self.z0neg = z0neg
        self.z1neg = z1neg

        self.obs_dim = self.z0.n_units
        self.hid_dim = self.z1.n_units

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.get()
        W1_msg = "W1:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1), jnp.linalg.norm(_W1)
        )
        _b1 = self.W1.biases.get()
        b1_msg = "b1:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_b1), jnp.amax(_b1), jnp.mean(_b1), jnp.linalg.norm(_b1)
        )
        _c1 = self.E1.biases.get()
        c1_msg = "c0:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_c1), jnp.amax(_c1), jnp.mean(_c1), jnp.linalg.norm(_c1)
        )
        return f"{W1_msg}\n{b1_msg}\n{c1_msg}"

    def viz_receptive_fields(self, fname, field_shape):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            fname: plot fname name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot
        """
        _W1 = self.W1.weights.get() #.T
        visualize([_W1], [field_shape], self.exp_dir + "/filters/{}".format(fname))

    def compute_energy(self, x):
        """
        Calculates the energy function E(x, z1) under this current harmonium/RBM set of parameters. (This is a function
        of input data-points and their respective latent codes assigned under this RBM.)

        Args:
            x: data patterns to calculate this model's energy over

        Returns:
            column vector of per-datapoint energy values, scalar energy assigned by this model to `x`
        """
        ## get latent codes for x under this RBM
        self.reset.run()
        self.clamp_input(x)
        self.advance_pos.run(t=0., dt=1.)  ## pos phase step (inference)
        ## given latent codes and x, calculate full energy
        return self._compute_energy(x, self.z1.s.get())

    def _compute_energy(self, x, z1): ## compute energy given data x & its respective latent codes z1 under this RBM
        z0 = x
        W1 = self.W1.weights.get() ## W
        hb1 = self.W1.biases.get() ## b
        vb0 = self.E1.biases.get() ## c
        ## calculate energy functional terms
        E_vb = jnp.matmul(z0, vb0.T) ## visible energy term
        E_hb = jnp.matmul(z1, hb1.T) ## hidden energy term
        E_W = jnp.sum(jnp.matmul(x, W1) * z1, axis=1, keepdims=True) ## quadratic energy term
        E = -E_vb - E_hb - E_W ## per-datapoint energy values
        Esum = jnp.sum(E) #jnp.sum(E, axis=0, keepdims=True) ## sum over batch dim
        return E, Esum

    def _update_via_CHL(self): ## contrastive Hebbian learning (CHL)
        self.calc_update.run(t=0., dt=1.)
        W = self.W1.weights.get()
        hb = self.W1.biases.get()
        vb = self.E1.biases.get()
        dWpos = self.W1.dWeights.get() ## get positive-phase grad
        dWneg = self.V1.dWeights.get() ## get negative-phase grad
        post_pos = self.W1.post.get()
        post_neg = self.V1.post.get()
        pre_pos = self.W1.pre.get()
        pre_neg = self.V1.pre.get()

        d_W, d_hb, d_vb = get_param_updates( ## combine & update via CHL recipe
            W, dWpos, dWneg, post_pos, post_neg, pre_pos, pre_neg, self.l1_lambda, self.l2_lambda
        )
        self.opt_params, theta = self.opt(
            self.opt_params,
            [self.W1.weights.get(), self.W1.biases.get(), self.E1.biases.get()],
            [d_W, d_hb, d_vb]
        )
        _W, _hb, _vb = theta

        self.W1.weights.set(_W)
        self.V1.weights.set(_W)
        self.E1.weights.set(_W.T) ## tie back feedback synapses
        self.W1.biases.set(_hb)
        self.V1.biases.set(_hb)
        self.E1.biases.set(_vb)

    def sample(
            self, dkey, n_steps=10, x_seed=None, thinning_point=0, burn_in=0, sample_buffer_maxlen=-1,
            n_samples=-1, verbose=False
    ):
        """
        Run block Gibbs sampling to obtain a chain of confabulations from the current parameters of this harmonium.

        Args:
            dkey: Jax key to seed sampler

            n_steps: how many steps to run sampler

            x_seed: data pattern(s) to initialize sampler from (Default: None);
                if None is provided, random uniform noise is used to seed this sampler

            thinning_point: if > 0, will thin the Markov chain this produces by extracting
                the sample every so many `thinning_point` steps

            burn_in: how many steps/samples of the Markov chain sample are discarded in order to minimize the effect
                of the initial values used to initialize the chain (useful for obtaining better-quality samples from
                the Gibbs sampler)

            sample_buffer_maxlen: maximum number of samples to store within the buffer returned by this method

            n_samples: if `x_seed` is None, this is used if > 0

            verbose: if True, this method will iteratively print to I/O its sampling progress

        Returns:
            array of samples from chain (n_steps x data-dimensionality)
        """
        self.reset.run() ## reset model

        _, *skey = random.split(dkey, 3)
        if x_seed is None: ## produce random noise to initialize chain
            _nsamp = int(max(n_samples, 1))
            p_eps = random.uniform(skey[0], shape=(_nsamp, self.obs_dim))
            eps = random.bernoulli(skey[1], p=p_eps, shape=p_eps.shape)
        else: ## seed chain w/ data (data-dependent initialization)
            eps = x_seed
        ## start chain off by running information thru positive-phase graph
        self.clamp_input(eps)
        self.advance_pos.run(t=0., dt=1.) ## seed Markov chain

        ## now continue down the chain by propagating through negative-phase graph
        x_samp = [] ## sample buffer
        for s in range(n_steps + burn_in):
            self.advance_neg.run(t=0., dt=1.) ## z1 -> z0neg -> z1neg
            x_s = self.z0neg.p.get()
            if s >= burn_in:
                if thinning_point > 0:
                    if s % thinning_point == 0:
                        x_samp.append(x_s)
                else:
                    x_samp.append(x_s)
            # self.reset.run()
            if self.is_meanfield:
                z1neg = self.z1neg.p.get()
                self.z1.p.set(z1neg)
            else:
                z1neg = self.z1neg.s.get()
                self.z1.s.set(z1neg)
            if sample_buffer_maxlen > 0:
                if len(x_samp) > sample_buffer_maxlen:
                    x_samp.pop(0)
            if verbose:
                print(f"\rGenerated {(s+1)} samples; len(buffer) = {len(x_samp)}", end="")
        if verbose:
            print()
        return x_samp

    def process(self, x, adapt_synapses=True):
        """
        Processes an observation (sensory stimulus pattern).

        Args:
            x: observed pattern(s) for RBM model to process
            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with contrastive Hebbian plasticity

        Returns:
            an array containing reconstruction vectors, scalar squared error,
                column vector of per-datapoint energies, scalar energy
        """
        self.reset.run()
        self.clamp_input(x)
        self.advance_pos.run(t=0., dt=1.) ## pos phase step
        self.advance_neg.run(t=0., dt=1.) ## neg phase step
        if adapt_synapses:
            self._update_via_CHL() ## make synaptic adjustments

        z1 = self.z1.s.get()
        E_x, E = self._compute_energy(x, z1)

        xR = self.z0neg.p.get() ## get mean-field reconstruction values        
        ## calc error (proxy to energy)
        error = jnp.sum(jnp.square(xR - x), axis=0, keepdims=True)
        error = jnp.sum(error)
        return xR, error, E_x, E
