from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
import time
from ngclearn.utils.model_utils import softmax
import sys

def tensorstats(tensor):
    if tensor is not None:
        _tensor = jnp.asarray(tensor)
        return {
            'mean': _tensor.mean(),
            'std': _tensor.std(),
            'mag': jnp.abs(_tensor).max(),
            'min': _tensor.min(),
            'max': _tensor.max(),
        }
    else:
        return {
            'mean': None,
            'std': None,
            'mag': None,
            'min': None,
            'max': None,
        }

def component_stats(component):
    maxlen = max(len(c) for c in component.compartments.keys()) + 5
    lines = f"[{component.__class__.__name__}] {component.name}\n"
    for comp_name, comp_value in component.compartments.items():
        stats = tensorstats(comp_value)
        line = [f"{k}: {v}" for k, v in stats.items()]
        line = ", ".join(line)
        lines += f"  {f'({comp_name})'.ljust(maxlen)}{line}\n"
    if hasattr(component, "weights"):
        stats = tensorstats(getattr(component, "weights"))
        line = [f"{k}: {v}" for k, v in stats.items()]
        line = ", ".join(line)
        lines += f"  {f'(weights)'.ljust(maxlen)}{line}\n"
    if hasattr(component, "biases"):
        stats = tensorstats(getattr(component, "biases"))
        line = [f"{k}: {v}" for k, v in stats.items()]
        line = ", ".join(line)
        lines += f"  {f'(biases)'.ljust(maxlen)}{line}\n"
    return lines

## SNN model co-routines
def load_model(model_dir, exp_dir="exp", model_name="snn_stdp", dt=3, T=100):
    _key = random.PRNGKey(time.time_ns())
    ## load circuit from disk
    circuit = Controller()
    circuit.load_from_dir(directory=model_dir)

    model = BFA_SNN(_key, in_dim=1, out_dim=1, save_init=False, dt=dt, T=T)
    model.circuit = circuit
    model.exp_dir = exp_dir
    model.model_dir = "{}/{}/custom".format(exp_dir, model_name)
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
    def __init__(self, dkey, in_dim, out_dim, hid_dim=1024, T=100, dt=0.25,
                 tau_m=20., exp_dir="exp", model_name="snn_bfa", save_init=True, **kwargs):
        self.exp_dir = exp_dir
        makedir(exp_dir)
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
        weightInit = ("gaussian", 0., 0.055) ## init synapses from centered Gaussian
        biasInit = ("constant", 0., 0.) ## init biases from zero values

        ### set up jax seeding
        dkey = random.PRNGKey(1234)
        dkey, *subkeys = random.split(dkey, 7) ## <-- chose 7 to get enough unique seeds for components

        ################################################################################
        ## Create/configure model and simulation object
        circuit = Controller()

        z0 = circuit.add_component("bernoulli", name="z0", n_units=in_dim, key=subkeys[0])
        W1 = circuit.add_component("hebbian", name="W1", shape=(in_dim, hid_dim),
                                   eta=1., wInit=weightInit, bInit=biasInit,
                                   signVal=-1., optim_type=optim_type, w_bound=0.,
                                   pre_wght=1., post_wght=eta1_w, is_nonnegative=False,
                                   key=subkeys[1])
        z1 = circuit.add_component("SLIF", name="z1", n_units=hid_dim, tau_m=tau_m, R_m=R_m,
                                   thr=v_thr, inhibit_R=0., sticky_spikes=True,
                                   refract_T=refract_T, thrGain=0., thrLeak=0.,
                                   thr_jitter=0., key=subkeys[2])
        W2 = circuit.add_component("hebbian", name="W2", shape=(hid_dim, out_dim),
                                   eta=1., wInit=weightInit, bInit=biasInit,
                                   signVal=-1., optim_type=optim_type, w_bound=0.,
                                   pre_wght=1., post_wght=eta2_w, is_nonnegative=False,
                                   key=subkeys[3])
        z2 = circuit.add_component("SLIF", name="z2", n_units=out_dim, tau_m=tau_m, R_m=R_m,
                                   thr=v_thr, inhibit_R=0., sticky_spikes=True,
                                   refract_T=refract_T, thrGain=0., thrLeak=0.,
                                   thr_jitter=0., key=subkeys[4])
        e2 = circuit.add_component("error", name="e2", n_units=out_dim)
        E2 = circuit.add_component("hebbian", name="E2", shape=(out_dim, hid_dim),
                                   eta=0., wInit=weightInit, bInit=None, w_bound=0.,
                                   is_nonnegative=False, key=subkeys[5])
        d1 = circuit.add_component("error", name="d1", n_units=hid_dim)

        ## wire up z0 to z1 via W1
        circuit.connect(z0.name, z0.outputCompartmentName(), W1.name, W1.inputCompartmentName())
        circuit.connect(W1.name, W1.outputCompartmentName(), z1.name, z1.inputCompartmentName())

        circuit.connect(z1.name, z1.outputCompartmentName(), W2.name, W2.inputCompartmentName())
        circuit.connect(W2.name, W2.outputCompartmentName(), z2.name, z2.inputCompartmentName())

        circuit.connect(e2.name, e2.derivMeanName(), E2.name, E2.inputCompartmentName())
        circuit.connect(E2.name, E2.outputCompartmentName(), d1.name, d1.targetName())

        circuit.connect(z2.name, z2.outputCompartmentName(), e2.name, e2.meanName())#, bundle="additive")
        #circuit.connect(zy.name, zy.outputCompartmentName(), e2.name, e2.targetName())

        ## wire relevant compartment statistics to synaptic cable z0_z1
        circuit.connect(z1.name, z1.surrogateCompartmentName(), d1.name, d1.modulatorName())
        circuit.connect(z0.name, z0.outputCompartmentName(), W1.name, W1.presynapticCompartmentName())
        circuit.connect(d1.name, d1.derivMeanName(), W1.name, W1.postsynapticCompartmentName())
        circuit.connect(z1.name, z1.outputCompartmentName(), W2.name, W2.presynapticCompartmentName())
        circuit.connect(e2.name, e2.derivMeanName(), W2.name, W2.postsynapticCompartmentName())

        ## make key commands known to model
        circuit.add_command("reset", command_name="reset",
                          component_names=[z0.name, W1.name, z1.name, W2.name, z2.name,
                                           e2.name, E2.name, d1.name],
                          reset_name="do_reset")
        circuit.add_command("advance", command_name="advance",
                          component_names=[z0.name, W1.name, z1.name, W2.name, z2.name,
                                           e2.name, E2.name, d1.name])
        circuit.add_command("evolve", command_name="evolve", component_names=[W1.name, W2.name])
        circuit.add_command("clamp", command_name="clamp_input",
                                 component_names=[z0.name], compartment=z0.inputCompartmentName(),
                                 clamp_name="x")
        circuit.add_command("clamp", command_name="clamp_target",
                                 component_names=[e2.name], compartment=e2.targetName(),
                                 clamp_name="y")
        circuit.add_command("save", command_name="save", component_names=[W1.name, W2.name, E2.name,
                                                                          z1.name, z2.name], directory_flag="dir")

        ## tell model the order in which to run automatic commands
        # circuit.add_step("clamp_input")
        circuit.add_step("advance")
        #circuit.add_step("evolve")
        ################################################################################

        if save_init == True: ## save JSON structure to disk once
            circuit.save_to_json(directory="exp", model_name=model_name)
        self.model_dir = "{}/{}/custom".format(exp_dir, model_name)
        if save_init == True:
            circuit.save(dir=self.model_dir) ## save current parameter arrays
        self.circuit = circuit # embed circuit to model construct

    def save_to_disk(self):
        """
        Saves current model parameter values to disk
        """
        self.circuit.save(dir=self.model_dir) ## save current parameter arrays

    def load_from_disk(self, model_directory="exp"):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit.load_from_dir(self, model_directory)

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.circuit.components.get("W1").weights
        _W2 = self.circuit.components.get("W2").weights
        msg = "W1.n = {}  W2.n = {}".format(jnp.linalg.norm(_W1), jnp.linalg.norm(_W2))
        return msg

    def process(self, obs, lab, adapt_synapses=True, collect_spike_train=False,
                label_dist_estimator="spikes", get_latent_rates=False,
                input_gain=0.25):
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
        _obs = _scale(obs, input_gain)
        rGamma = 1.
        _S = []
        if get_latent_rates == True:
            _S = jnp.zeros((obs.shape[0], self.circuit.components["z1"].n_units))
        yMu = jnp.zeros((obs.shape[0], self.circuit.components["z2"].n_units))
        yCnt = yMu + 0
        self.circuit.reset(do_reset=True)
        T_learn = 0.
        for ts in range(1, self.T):
            self.circuit.clamp_input(_obs) #x=inp)
            self.circuit.clamp_target(lab) #y=lab
            self.circuit.runCycle(t=ts*self.dt, dt=self.dt)
            curr_t = ts * self.dt ## track current time
            if adapt_synapses == True:
                if curr_t > self.burnin_T:
                    self.circuit.evolve(t=self.T, dt=self.dt)
            yCnt = _add(self.circuit.components["z2"].spikes, yCnt)
            ## estimate output distribution
            if curr_t > self.burnin_T:
                T_learn += 1.
                if label_dist_estimator == "current":
                    yMu = _add(self.circuit.components["z2"].current, yMu)
                elif label_dist_estimator == "voltage":
                    yMu = _add(self.circuit.components["z2"].voltage, yMu)
                else: # label_dist_estimator == "spike":
                    yMu = _add(self.circuit.components["z2"].spikes, yMu)
            ## collect internal/hidden spikes at t
            if get_latent_rates == True:
                _S = _add(_S, self.circuit.components["z1"].spikes)
            else:
                _S.append(self.circuit.components["z1"].spikes)

            ######### Logging/Model Matching #############
            # if ts == 2:
            #     print(f"{component_stats(self.circuit.components.get('z0'))}")
            #     print(f"{component_stats(self.circuit.components.get('W1'))}")
            #     print(f"{component_stats(self.circuit.components.get('z1'))}")
            #     print(f"{component_stats(self.circuit.components.get('W2'))}")
            #     print(f"{component_stats(self.circuit.components.get('z2'))}")
            #     print(f"{component_stats(self.circuit.components.get('e2'))}")
            #     print(f"{component_stats(self.circuit.components.get('E2'))}")
            #     print(f"{component_stats(self.circuit.components.get('d1'))}")
            #     sys.exit(0)
            ##############################################

        _yMu = softmax(yMu/T_learn) #self.T) ## estimate approximate label distribution
        if get_latent_rates == True:
            _S = (_S * rGamma)/self.T
        return _S, _yMu, yCnt

@jit
def _scale(x, factor):
    ## small jit co-routine used by BFA-SNN process function
    return x * factor
