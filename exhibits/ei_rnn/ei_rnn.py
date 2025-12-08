from ngclearn.utils.io_utils import makedir
#from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess, JointProcess
from custom.inputCell import InputCell
from ngclearn.components.neurons.graded.leakyNoiseCell import LeakyNoiseCell
from ngclearn.components.neurons.graded.gaussianErrorCell import GaussianErrorCell as ErrorCell
from ngclearn.components.synapses.staticSynapse import StaticSynapse
from custom.tracedHebbianSynapse import TracedHebbianSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.utils.model_utils import relu
from ngcsimlib.operations import Summation, Product
from ngcsimlib.global_state import stateManager

@jit
def noisify_data(dkey, obs_t, u0, alpha, sigma_in): ## applies Eqn 10/11 in (Song et al., 2016) to data obs(t)
    _, *subkeys = random.split(dkey, 3)
    eps = random.normal(subkeys[0], shape=obs_t.shape)
    u_task_t = obs_t
    u_t = relu(u0 + u_task_t + (1. / alpha) * jnp.sqrt(2. * alpha * jnp.square(sigma_in)) * eps)
    return u_t

class EI_RNN(): ## continuous-time excitatory-inhibitory (EI) recurrent neural network (RNN)
    """
    Structure for constructing an excitatory-inhibitory recurrent neural network (EI-RNN) 
    as in:

    | Song, H. F., Yang, G. R., & Wang, X. J. (2016). Training excitatory-inhibitory recurrent 
    | neural networks for cognitive tasks: a simple and flexible framework. PLoS computational 
    | biology, 12(2), e1004792.

    This RNN model is constrained to only allow non-negative synaptic weight connections (respecting Dale's principle
    via pre-determined positive/negative signs) and to ensure that its hidden state is a population consisting of 80%
    excitatory neurons and 20% inhibitory neurons (respecting the biological 4:1 ratio of excitation to inhibition).

    Args:
        dkey: JAX seeding key
        obs_dim: number of input units
        hid_dim: number of hidden state units (80% excitatory, 20% inhibitory)
        out_dim: number of readout units
        sigma_in: input noise scale; if > 0, then inputs will be treated as part of a noisy time-series
            (i.e., as per Eqn 11 in Song et al., 2016)
        exp_dir: string indicating directory to save model/analysis results
        loadDir: <unused>
    """

    def __init__(
            self, dkey, obs_dim=1, hid_dim=2, out_dim=1, sigma_in=0., exp_dir="exp", model_name="ei_rnn",
            loadDir=None, **kwargs
    ):
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        makedir(exp_dir + "/analysis")

        self.dkey = dkey
        ## EI-RNN (meta-)parameters
        exc_pct = 0.8 ## ensures we respect 4/1 ratio for exc/inh
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        n_exc = int(hid_dim * exc_pct) ## num excitatory neurons
        n_inh = hid_dim - n_exc ## num inhibitory neurons
        
        tau_x = 100. ## membrane time constant (ms)
        act_fx = "relu" ## post-membrane activation
        sigma_pre = 0.1 ## pre-rectification noise std-dev
        init_sigma = 0.02
        self.dt = 10. ## integration time constant (ms)
        ## general optimization (meta-)parameters
        eta = 0.0002
        optim_type = "adam" # "sgd"
        tau_elg = 40. #15. #30. #20. # ms
        ## Dale's law will be respected via non-negativity enforcement (after synaptic updates) and
        ## pre-determined signs (according to the 4/1 ratio set-up)

        ## noisy input parameters (if sigma_in > 0)
        ### u(t) = relu(u0 + obs(t) + (1./alpha) * jnp.sqrt(2. * alpha * jnp.square(sigma_in)) * eps)
        ### where eps ~ N(0, 1)
        self.sigma_in = sigma_in
        self.u0 = 0. ## input baseline / shift value
        self.alpha = self.dt/tau_x
    
        self.dkey, *subkeys = random.split(self.dkey, 10)
        with Context("Circuit") as self.circuit:
            ## set up neural populations (input, hidden, readout)
            z0 = InputCell("z0", n_units=obs_dim) ## input state (driven by data)
            z1_prev = InputCell("z1_prev", n_units=hid_dim) ## last copy of z1 state
            z1 = LeakyNoiseCell( ## hidden state
                "z1", n_units=hid_dim, tau_x=tau_x, act_fx=act_fx, sigma_pre=sigma_pre, leak_scale=1.,
                key=subkeys[0]
            )
            o1 = InputCell("o1", n_units=out_dim) ## output state/projection
            e1o = ErrorCell("e1o", n_units=out_dim) ## readout error neurons
            d1 = InputCell("d1", n_units=hid_dim) ## teaching signal for z1

            ## set up synaptic connections between populations
            W1 = TracedHebbianSynapse( ## input-to-hidden connections
                "W1", shape=(obs_dim, hid_dim), eta=eta, weight_init=dist.uniform(low=0., high=init_sigma),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[1],
                tau_elg=tau_elg
            )
            V1signs = jnp.concat((jnp.ones((n_exc, 1)), -jnp.ones((n_inh, 1))), axis=0)
            self.V1mask = jnp.ones((hid_dim, hid_dim)) * V1signs
            V1 = TracedHebbianSynapse( ## recurrent connections
                "V1", shape=(hid_dim, hid_dim), eta=eta, weight_init=dist.uniform(low=0., high=init_sigma), 
                bias_init=None, w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[2],
                mask=self.V1mask, tau_elg=tau_elg
            )
            W1o = TracedHebbianSynapse( ## output/readout connections
                "W1o", shape=(hid_dim, out_dim), eta=eta, weight_init=dist.uniform(low=0., high=init_sigma),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[3],
                tau_elg=tau_elg
            )
            E1o = StaticSynapse( ## feedback connections
                "E1o", shape=(out_dim, hid_dim), weight_init=dist.uniform(low=0., high=init_sigma), resist_scale=1.,
                key=subkeys[4]
            )
            E1o.weights.set(W1o.weights.get().T) ## feedback = transpose(readout weights)

            ## wire inputs to hidden layer
            z0.inputs >> W1.inputs 
            W1.outputs >> z1.j_input ## input stimulus
            z1_prev.outputs >> V1.inputs
            V1.outputs >> z1.j_recurrent ## recurrent stimulus
            ## wire hidden layer to output layer
            z1.r >> W1o.inputs
            W1o.outputs >> o1.inputs
            o1.outputs >> e1o.mu
            ## wire output error neurons to hidden
            e1o.dmu >> E1o.inputs
            Product(E1o.outputs, z1.r_prime) >> d1.inputs

            ## wire up local update rules
            ### input-to-hidden update rule
            z0.outputs >> W1.pre
            d1.outputs >> W1.post
            ### recurrent update rule
            z1_prev.outputs >> V1.pre
            d1.outputs >> V1.post
            ### output/readout update rule
            z1.r >> W1o.pre
            e1o.dmu >> W1o.post

            ## advance-states function
            self.advance = (MethodProcess(name="advance")
                            >> z0.advance_state
                            >> z1_prev.advance_state
                            >> W1.advance_state
                            >> V1.advance_state
                            >> z1.advance_state
                            >> W1o.advance_state
                            >> o1.advance_state
                            >> e1o.advance_state
                            >> E1o.advance_state
                            >> d1.advance_state)
            
            ## evolve/adaptation function
            self.calc_update = (MethodProcess(name="calc_update")
                                >> W1.calc_update
                                >> W1o.calc_update
                                >> V1.calc_update)
            self.evolve = (MethodProcess(name="evolve")
                           >> W1.evolve
                           >> W1o.evolve
                           >> V1.evolve)

            ## reset-to-baseline function
            self.reset = (MethodProcess(name="reset")
                          >> z0.reset
                          >> z1_prev.reset
                          >> W1.reset
                          >> V1.reset
                          >> z1.reset
                          >> W1o.reset
                          >> o1.reset
                          >> e1o.reset
                          >> E1o.reset
                          >> d1.reset)

    def clamp(self, obs):
        """
        Clamps input patterns (at time t) x to this model.

        Args:
            obs: input patterns (at time t) to clamp
        """
        z0 = self.circuit.get_components("z0")
        z0.inputs.set(obs) 

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only: ## this condition allows to only write actual parameter values w/in components to disk
            W1, V1, W1o, E1o = self.circuit.get_components("W1", "V1", "W1o", "E1o")
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            W1.save(model_dir)
            V1.save(model_dir)
            W1o.save(model_dir)
            E1o.save(model_dir)
        else: ## this saves the whole model form (JSON structure as well as parameter values)
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        #self.advance = self.circuit.get_objects("advance_process", objectType="process")
        processes = self.circuit.get_objects_by_type("process") ## obtain all saved processes within this context
        self.advance = processes.get("advance")
        self.calc_update = processes.get("calc_update")
        self.evolve = processes.get("evolve")
        self.reset = processes.get("reset")

        z0, z1, o1 = self.circuit.get_components("z0", "z1", "o1")
        self.obs_dim = z0.n_units
        self.hid_dim = z1.n_units
        self.out_dim = o1.n_units

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of synaptic parameters/biases
        """
        W1, V1, W1o, E1o = self.circuit.get_components("W1", "V1", "W1o", "E1o")

        _W1 = W1.weights.get()
        W1_msg = "W1:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1), jnp.linalg.norm(_W1)
        )
        _b1 = W1.biases.get()
        b1_msg = "b1:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_b1), jnp.amax(_b1), jnp.mean(_b1), jnp.linalg.norm(_b1)
        )
        _W1 = W1o.weights.get()
        W1o_msg = "W1o:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1), jnp.linalg.norm(_W1)
        )
        _b1 = W1o.biases.get()
        b1o_msg = "b1o:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_b1), jnp.amax(_b1), jnp.mean(_b1), jnp.linalg.norm(_b1)
        )

        _W1 = V1.weights.get()
        V1_msg = "V1:  min {:.4f} ;  max {:.4f}  mu {:.4f} ;  norm {:.4f}".format(
            jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1), jnp.linalg.norm(_W1)
        )
        return f"{W1_msg}\n{b1_msg}\n{W1o_msg}\n{b1o_msg}\n{V1_msg}"

    def reset_vals(self):
        """
        Sets model's internal states to baseline values / initial conditions
        """
        self.reset.run()

    def process(self, obs_seq, targ_seq=None, adapt_synapses=False): ## process temporal sequence
        """
        Process input observation sequence (with the potential aim of making predictions of the form
        p(o_{t+1} | o_{<=t}) (i.e., predict t+1 given the input history so far).

        Args:
            obs_seq: observation sequence (i.e., o(t) where t=1, 2, 3,...,T)
            targ_seq: target value sequence (must align one-to-one to observation sequence and should be shifted
                forward in time by one time-step, i.e., o(t) where t=2, 3, 4,...,T)
            adapt_synapses: if True, synapses will be adjusted after negative phase
                statistics are obtained; (Default: False)

        Returns:
            sequence of hidden variables, sequence of output variables, sequence loss (scalar)
        """
        T = obs_seq.shape[0] ## get number of timesteps in time-series
        _obs_seq = obs_seq
        if self.sigma_in > 0.: ## apply input noise if configured
            self.dkey, *skey = random.split(self.dkey, 3)
            _obs_seq = noisify_data(skey[0], _obs_seq, self.u0, self.alpha, self.sigma_in)

        z0, z1_prev, z1, o1, e1o, d1 = self.circuit.get_components("z0", "z1_prev", "z1", "o1", "e1o", "d1")
        W1, V1, W1o, E1o = self.circuit.get_components("W1", "V1", "W1o", "E1o")

        self.reset.run()
        hids = []
        outs = []
        loss_seq = 0. ## sequence loss -> L_seq = sum_t [ L(t) ]
        for t in range(T):
            obs_t = _obs_seq[t:t+1, :] ## o(t)
            self.clamp(obs_t)
            if targ_seq is not None:
                targ_t = targ_seq[t:t+1, :]
                e1o.target.set(targ_t) ## clamp error neurons to target
            self.advance.run(t=t*self.dt, dt=self.dt) ## take inference step forward
            loss_seq += e1o.L.get()
            if adapt_synapses:
                self.evolve.run(t=1., dt=1.) ## adjust synapses
                ## enforce non-negativity
                W1.weights.set(jnp.maximum(0., W1.weights.get()))
                W1o.weights.set(jnp.maximum(0., W1o.weights.get()))
                V1.weights.set(jnp.maximum(0., V1.weights.get()))
                E1o.weights.set(W1o.weights.get().T)  ## tie feedback to forward syn

            z1_prev.inputs.set( z1.r.get() ) ## set prior state memory 
            ## collect statistics
            hids.append( z1.r.get() ) ## get z(t)
            outs.append( o1.outputs.get() )
        hids = jnp.concat(hids, axis=0)
        outs = jnp.concat(outs, axis=0)
        return hids, outs, loss_seq
