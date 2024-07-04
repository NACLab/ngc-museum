from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.components.input_encoders.bernoulliCell import BernoulliCell
from ngclearn.components.synapses.hebbian.eventSTDPSynapse import EventSTDPSynapse
from ngclearn.components.other.expKernel import ExpKernel
from ngclearn.components.neurons.spiking.WTASCell import WTASCell
import ngclearn.utils.weight_distribution as dist

class SNN():
    """
    Structure for constructing a spiking neural model adapted via
    exponential spike-timing-dependent plasticity (exp-stdp).

    | Node Name Structure:
    | z0 -(W1)-> z1 <-z1(t-dt)
    | Note: W1 = STDP-adapted synapses, z1 is implicitly recurrent (lateral inhibition)

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        hid_dim: dimensionality of the representation layer of neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with
    """
    # Define Functions
    def __init__(self, dkey, in_dim, hid_dim=100, T=200, dt=1., exp_dir="exp",
                 model_name="snn_evstdp", load_dir=None, **kwargs):
        dkey, *subkeys = random.split(dkey, 10)
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        ## meta-parameters for model dynamics
        self.T = T
        self.dt = dt
        eta_w = 0.0055
        thrBase = 0.2
        thr_gain = 0.002
        tau_m = 100.

        if load_dir is not None:
            ## build from disk
            self.load_from_disk(load_dir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = BernoulliCell("z0", n_units=in_dim, key=subkeys[0])
                self.k0 = ExpKernel("k0", n_units=in_dim, tau_w=0.5, nu=4., dt=dt, key=subkeys[1])
                self.W1 = EventSTDPSynapse("W1", shape=(in_dim, hid_dim), eta=eta_w,
                                           lmbda=0.01, w_bound=1., presyn_win_len=3.,
                                           weight_init=dist.uniform(0.025, 0.8), resist_scale=1.,
                                           key=subkeys[2])
                self.z1 = WTASCell("z1", n_units=hid_dim, tau_m=tau_m, resist_m=1.,
                                   thrBase=thrBase, thr_gain=thr_gain, refract_time=5.,
                                   thr_jitter=0.055, key=subkeys[3])

                ## wire z0 to z1e via W1 and z1i to z1e via W1ie
                self.k0.inputs << self.z0.outputs
                self.W1.inputs << self.k0.epsp
                self.z1.j << self.W1.outputs
                # wire relevant compartment statistics to synaptic cable W1
                self.W1.pre_tols << self.z0.tols #self.W1.preSpike << self.z0.outputs
                self.W1.postSpike << self.z1.s

                reset_cmd, reset_args = self.circuit.compile_by_key(
                                            self.z0, self.k0, self.z1,
                                            self.W1,
                                            compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_by_key(
                                                self.W1,
                                                self.z0, self.k0, self.z1,
                                                compile_key="advance_state")
                evolve_cmd, evolve_args = self.circuit.compile_by_key(self.W1, compile_key="evolve")
                self.dynamic()

    def dynamic(self):## create dynamic commands for circuit
        W1, z0, z1 = self.circuit.get_components("W1", "z0", "z1")
        self.W1 = W1
        self.z0 = z0
        self.z1 = z1

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")

        @Context.dynamicCommand
        def clamp(x):
            z0.inputs.set(x)

        @scanner
        def process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(
                compartment_values, t=_t, dt=_dt)
            compartment_values = self.circuit.evolve(
                compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.z1.s.path]

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only is True:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.z1.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name) ## save current parameter arrays

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory)
            self.dynamic()

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.value
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
        _W1 = self.W1.weights.value
        visualize([_W1], [field_shape], self.exp_dir + "/filters/{}".format(fname))

    def process(self, obs, adapt_synapses=True, collect_spike_train=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T. Note that the observed pattern will be converted
        to a Poisson spike train with maximum frequency of 63.75 Hertz.

        Note that this model assumes batch sizes of one (online learning).

        Args:
            obs: observed pattern to have spiking model process

            adapt_synapses: if True, synaptic efficacies will be adapted in
                accordance with trace-based spike-timing-dependent plasticity

            collect_spike_train: if True, will store an T-length array of spike
                vectors for external analysis

        Returns:
            an array containing spike vectors (will be empty; length = 0 if
                collect_spike_train is False)
        """
        batch_dim = obs.shape[0]
        assert batch_dim == 1 ## batch-length must be one for DC-SNN

        self.circuit.reset()
        self.circuit.clamp(obs)
        out = self.circuit.process(jnp.array([[self.dt*i, self.dt]
                                   for i in range(self.T)]))
        return out
