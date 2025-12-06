# %%

from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
import time

# from ngclearn.utils.model_utils import scanner
# from ngcsimlib.compilers import wrap_command
# from ngcsimlib.context import Context
# from ngclearn.utils import JaxProcess
from ngclearn import Context, MethodProcess, JointProcess
from ngclearn.utils.model_utils import softmax
from ngclearn.components import (GaussianErrorCell, SLIFCell, BernoulliCell,
                                 HebbianSynapse, StaticSynapse)
# import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.distribution_generator import DistributionGenerator


## SNN model co-routines
def load_model(exp_dir="exp", dt=3, T=100):
    _key = random.PRNGKey(time.time_ns())
    model = BFA_SNN(_key, in_dim=1, out_dim=1, save_init=False, dt=dt, T=T)
    model.load_from_disk(exp_dir)
    return model

@jit
def _add(x, y): ## jit-i-fied vector-matrix add
    return x + y

class BFA_SNN():
    """
    Structure for constructing the spiking neural model proposed in:

    Samadi, Arash, Timothy P. Lillicrap, and Douglas B. Tweed. "Deep learning with
    dynamic spiking neurons and fixed feedback weights." Neural computation 29.3
    (2017): 578-602.

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        hid_dim: dimensionality of the representation layer of neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        tau_m: membrane time constant (for hidden and output layers of LIFs)

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with

        save_init: save model at initialization/first configuration time (Default: True)
    """
    # Define Functions
    def __init__(self, dkey, in_dim=1, out_dim=1, hid_dim=1024, T=100, dt=0.25,
                 tau_m=20., exp_dir="exp", model_name="snn_bfa", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        #makedir("{}/{}".format(exp_dir, model_name))
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        self.T = T ## num discrete time steps to simulate
        self.dt = dt ## integration time constant
        self.burnin_T = 20. * dt # ms ## num time steps where learning is inhibited (only applies to training)

        R_m = 1. ## input resistance (to sLIF cells)
        optim_type = "sgd"
        ## layer-wise learning rates (as per Samadi et al., 2017)
        eta1_w = 1.0/(in_dim * 1.0) ## z1's learning rate
        eta2_w = 1.0/(hid_dim * 1.0) ## z2's learning rate

        v_thr = 0.4
        refract_T = 1.
        weightInit = DistributionGenerator.gaussian(0., 0.055) ## init synapses from centered Gaussian
        print(f"[BFA_SNN.__init__] Weight init: {weightInit}")
        biasInit = DistributionGenerator.constant(0.) ## init biases from zero values

        ### set up jax seeding
        #dkey = random.PRNGKey(1234)
        dkey, *subkeys = random.split(dkey, 7) ## <-- chose 7 to get enough unique seeds for components

        ################################################################################
        ## Create/configure model and simulation object
        if loadDir is not None:
            ## build from disk
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = BernoulliCell(name="z0", n_units=in_dim, key=subkeys[0])
                self.W1 = HebbianSynapse(name="W1", shape=(in_dim, hid_dim),
                                         eta=1., weight_init=weightInit, bias_init=biasInit,
                                         sign_value=-1., optim_type=optim_type, w_bound=0.,
                                         pre_wght=1., post_wght=eta1_w, is_nonnegative=False,
                                         key=subkeys[1])
                self.z1 = SLIFCell(name="z1", n_units=hid_dim, tau_m=tau_m, resist_m=R_m,
                                   thr=v_thr, resist_inh=0., sticky_spikes=True,
                                   refract_time=refract_T, thr_gain=0., thr_leak=0.,
                                   thr_jitter=0., key=subkeys[2])
                self.W2 = HebbianSynapse(name="W2", shape=(hid_dim, out_dim),
                                         eta=1., weight_init=weightInit, bias_init=biasInit,
                                         sign_value=-1., optim_type=optim_type, w_bound=0.,
                                         pre_wght=1., post_wght=eta2_w, is_nonnegative=False,
                                         key=subkeys[3])
                self.z2 = SLIFCell(name="z2", n_units=out_dim, tau_m=tau_m, resist_m=R_m,
                                   thr=v_thr, resist_inh=0., sticky_spikes=True,
                                   refract_time=refract_T, thr_gain=0., thr_leak=0.,
                                   thr_jitter=0., key=subkeys[4])
                self.e2 = GaussianErrorCell(name="e2", n_units=out_dim)
                self.E2 = StaticSynapse(name="E2", shape=(out_dim, hid_dim),
                                        weight_init=weightInit, bias_init=None,
                                        key=subkeys[5])
                self.d1 = GaussianErrorCell(name="d1", n_units=hid_dim)


                ## wire up z0 to z1 via W1
                self.z0.outputs >> self.W1.inputs
                self.W1.outputs >> self.z1.j

                self.z1.s >> self.W2.inputs
                self.W2.outputs >> self.z2.j
                self.z2.s >> self.e2.mu

                self.e2.dmu >> self.E2.inputs
                self.E2.outputs >> self.d1.target

                ## wire relevant compartment statistics to synaptic cable z0_z1
                self.z1.surrogate >> self.d1.modulator
                self.z0.outputs >> self.W1.pre
                self.d1.dmu >> self.W1.post
                self.z1.s >> self.W2.pre
                self.e2.dmu >> self.W2.post

                # Create Process objects for reset, advance, and evolve
                self.reset_process = (MethodProcess(name="reset_process")
                                >> self.z0.reset
                                >> self.W1.reset
                                >> self.z1.reset
                                >> self.W2.reset
                                >> self.z2.reset
                                >> self.e2.reset
                                >> self.E2.reset
                                >> self.d1.reset)

                self.advance_process = (MethodProcess(name="advance_process")
                                  >> self.z0.advance_state
                                  >> self.W1.advance_state
                                  >> self.z1.advance_state
                                  >> self.W2.advance_state
                                  >> self.z2.advance_state
                                  >> self.e2.advance_state
                                  >> self.E2.advance_state
                                  >> self.d1.advance_state)

                self.evolve_process = (MethodProcess(name="evolve_process")
                                 >> self.W1.evolve
                                 >> self.W2.evolve)

                # self._dynamic(processes)

    def clamp(self, x, y):
        self.z0.inputs.set(x)
        self.e2.target.set(y)

    # def _dynamic(self, processes):
    #     @scanner
    #     def process(compartment_values, args):
    #         _t, _dt = args
    #         compartment_values = advance_process.pure(compartment_values, t=_t, dt=_dt)
    #         compartment_values = evolve_process.pure(compartment_values, t=_t, dt=_dt)
    #         return compartment_values, (compartment_values[self.z1.s.path],
    #                                     compartment_values[self.z2.s.path])


    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.z1.save(model_dir)
            self.W2.save(model_dir)
            self.z2.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name, overwrite=True) ## save current parameter arrays

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            # processes = (self.circuit.reset_process, self.circuit.advance_process, self.circuit.evolve_process)
            # self._dynamic(processes)

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.value
        _W2 = self.W2.weights.value
        msg = "W1.n = {}  W2.n = {}".format(jnp.linalg.norm(_W1), jnp.linalg.norm(_W2))
        return msg

    def process(
            self, obs, lab, adapt_synapses=True, collect_spike_train=False, label_dist_estimator="spikes", 
            get_latent_rates=False, input_gain=0.25
    ):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T. Note that the observed pattern will be converted
        to a Poisson spike train with maximum frequency of 63.75 Hertz.

        Args:
            obs: observed pattern(s) to have spiking model process

            lab: label pattern(s) to have spiking model use (used only if
                `adapt_synapses = True`)

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with trace-based spike-timing-dependent plasticity

            collect_spike_train: if True, will store an T-length array of spike
                vectors for external analysis

            label_dist_estimator: string indicating what type of model data to
                estimate label distributions from; "current" (the Default) means use the
                electrical current at `t`, "voltage" means use the voltage
                at `t`, and "spike" means use the spikes/potentials at `t`

            get_latent_rates: if True, then first output of this funciton will
                instead be a compressed matrix containing rate codes
                (one per sample) instead of being a list of spike batches

            input_gain: gain factor (0 < `input_gain` <= 1.) to scale raw
                input data by before spike production (Default: 0.25)

        Returns:
            an array containing spike vectors (will be empty; length = 0 if
                collect_spike_train is False), estimated label distribution
        """
        ## check and configure batch size
        # for node in self.nodes:
        #     node.batch_size = obs.shape[0]

        ## now run the model with configured batch size
        _obs = _scale(obs, input_gain)
        rGamma = 1.
        _S = []
        if get_latent_rates == True:
            _S = jnp.zeros((obs.shape[0], self.z1.n_units))
        yMu = jnp.zeros((obs.shape[0], self.z2.n_units))
        yCnt = yMu + 0
        self.reset_process.run()
        T_learn = 0.
        for ts in range(1, self.T):
            self.clamp(_obs, lab)
            self.advance_process.run(t=ts*self.dt, dt=self.dt)
            curr_t = ts * self.dt ## track current time

            if adapt_synapses == True:
                if curr_t > self.burnin_T:
                    self.evolve_process.run(t=ts*self.dt, dt=self.dt)
            yCnt = _add(self.z2.s.get(), yCnt)

            ## estimate output distribution
            if curr_t > self.burnin_T:
                T_learn += 1.
                if label_dist_estimator == "current":
                    yMu = _add(self.z2.j.get(), yMu)
                elif label_dist_estimator == "voltage":
                    yMu = _add(self.z2.v.get(), yMu)
                else:
                    yMu = _add(self.z2.s.get(), yMu)
            ## collect internal/hidden spikes at t
            if get_latent_rates == True:
                _S = _add(_S, self.z1.s.get())
            else:
                _S.append(self.z1.s.get())

        _yMu = softmax(yMu/T_learn) ## estimate approximate label distribution
        if get_latent_rates == True:
            _S = (_S * rGamma)/self.T

        return _S, _yMu, yCnt

@jit
def _scale(x, factor):
    ## small jit co-routine used by BFA-SNN process function
    return x * factor
