#from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
#from jax.lax import scan
from ngclearn.utils.model_utils import scanner
import time

from ngcsimlib.compilers import compile_command, wrap_command

#from ngcsimlib.compartment import All_compartments
from ngcsimlib.context import Context
from ngcsimlib.commands import Command
from ngcsimlib.operations import summation
from ngclearn.components.other.varTrace import VarTrace
from ngclearn.components.input_encoders.poissonCell import PoissonCell
from ngclearn.components.neurons.spiking.LIFCell import LIFCell
from ngclearn.components.synapses.hebbian.traceSTDPSynapse import TraceSTDPSynapse
from ngclearn.components.synapses.hebbian.hebbianSynapse import HebbianSynapse
from ngclearn.utils.model_utils import normalize_matrix

## SNN model co-routines
def load_model(model_dir, exp_dir="exp", model_name="snn_stdp", dt=1., T=200, in_dim=1):
    _key = random.PRNGKey(time.time_ns())
    ## load circuit from disk
    #circuit = Controller()
    #circuit.load_from_dir(directory=model_dir)

    model = DC_SNN(_key, in_dim=in_dim, save_init=False, dt=dt, T=T)
    #model.circuit = circuit
    model.exp_dir = exp_dir
    model.model_dir = "{}/{}/custom".format(exp_dir, model_name)
    model.load_from_disk(model.model_dir)
    return model

class DC_SNN():
    """
    Structure for constructing the spiking neural model proposed in:

    Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit recognition
    using spike-timing-dependent plasticity." Frontiers in computational
    neuroscience 9 (2015): 99.

    | Node Name Structure:
    | z0 -(W1)-> z1e <-(W1ie)- z1i ; z1i <-(W1ei)- z1e
    | Note: W1 = STDP-adapted synapses, W1ie and W1ei are fixed

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        hid_dim: dimensionality of the representation layer of neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with

        save_init: save model at initialization/first configuration time (Default: True)
    """
    # Define Functions
    def __init__(self, dkey, in_dim, hid_dim=100, T=200, dt=1., exp_dir="exp",
                 model_name="snn_stdp", save_init=True, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        makedir("{}/{}".format(exp_dir, model_name))
        makedir(exp_dir + "/filters")
        makedir(exp_dir + "/raster")

        self.T = T #250 # ms (num discrete time steps to simulate)
        self.dt = dt # ms (integration time constant)
        tau_m_e = 100.500896468 # ms (excitatory membrane time constant)
        tau_m_i = 100.500896468 # ms (inhibitory membrane time constant)
        tau_tr= 20. # ms (trace time constant)

        ## STDP hyper-parameters
        Aplus = 1e-2 ## LTD learning rate (STDP); nu1
        Aminus = 1e-4 ## LTD learning rate (STDP); nu0
        self.wNorm = 78.4 ## post-stimulus window norm constraint to apply to synapses

        dkey, *subkeys = random.split(dkey, 10)

        with Context("Circuit") as circuit:
            self.z0 = PoissonCell("z0", n_units=in_dim, max_freq=63.75, key=subkeys[0])
            self.W1 = TraceSTDPSynapse("W1", shape=(in_dim, hid_dim), eta=1.,
                                       Aplus=Aplus, Aminus=Aminus, wInit=("uniform", 0.0, 0.3),
                                       preTrace_target=0., key=subkeys[1])
            self.z1e = LIFCell("z1e", n_units=hid_dim, tau_m=tau_m_e, R_m=1., thr=-52.,
                               v_rest=-65., v_reset=-60., tau_theta=1e7, theta_plus=0.05,
                               refract_T=5., one_spike=True, key=subkeys[2])
            self.z1i = LIFCell("z1i", n_units=hid_dim, tau_m=tau_m_i, R_m=1., thr=-40.,
                               v_rest=-60., v_reset=-45., tau_theta=0., refract_T=5.,
                               one_spike=False, key=subkeys[3])

            # ie -> inhibitory to excitatory; ei -> excitatory to inhibitory
            #       (eta = 0 means no learning)
            self.W1ie = HebbianSynapse("W1ie", shape=(hid_dim, hid_dim), eta=0.,
                                       wInit=("hollow", -120., 0.), w_bound=0.,
                                       key=subkeys[4])
            self.W1ei = HebbianSynapse("W1ei", shape=(hid_dim, hid_dim), eta=0.,
                                       wInit=("eye", 22.5, 0), w_bound=0.,
                                       key=subkeys[5])
            self.tr0 = VarTrace("tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="exp",
                                a_delta=0., key=subkeys[6])
            self.tr1 = VarTrace("tr1", n_units=hid_dim, tau_tr=tau_tr, decay_type="exp",
                                a_delta=0., key=subkeys[7])

            ## wire z0 to z1e via W1 and z1i to z1e via W1ie
            self.W1.inputs << self.z0.outputs
            self.W1ie.inputs << self.z1i.s
            self.z1e.j << summation(self.W1.outputs, self.W1ie.outputs)
            # wire z1e to z1i via W1ie
            self.W1ei.inputs << self.z1e.s
            self.z1i.j << self.W1ei.outputs
            # wire cells z0 and z1e to their respective traces
            self.tr0.inputs << self.z0.outputs
            self.tr1.inputs << self.z1e.s
            # wire relevant compartment statistics to synaptic cable W1
            self.W1.preTrace << self.tr0.trace
            self.W1.preSpike << self.z0.outputs
            self.W1.postTrace << self.tr1.trace
            self.W1.postSpike << self.z1e.s

            reset_cmd, reset_args = circuit.compile_command_key(
                                        self.z0, self.z1e, self.z1i,
                                        self.tr0, self.tr1,
                                        self.W1, self.W1ie, self.W1ei,
                                    compile_key="reset")
            advance_cmd, advance_args = circuit.compile_command_key(self.W1, self.W1ie, self.W1ei,
                                                    self.z0, self.z1e, self.z1i,
                                                    self.tr0, self.tr1,
                                         compile_key="advance_state")
            evolve_cmd, evolve_args = circuit.compile_command_key(self.W1, compile_key="evolve")

            circuit.add_command(wrap_command(jit(reset_cmd)), name="reset")

            @scanner
            def process(compartment_values, args):
                t = args[0]
                dt = args[1]
                compartment_values = circuit.advance_state(compartment_values, t, dt)
                compartment_values = circuit.evolve(compartment_values, t, dt)
                return compartment_values, compartment_values[self.z1e.s.path]

            ## some helper dynamic commands
            @circuit.dynamicCommand
            def norm():
                self.W1.weights.set(normalize_matrix(self.W1.weights.value, self.wNorm, order=1, axis=0))

            @circuit.dynamicCommand
            def clamp(x):
                self.z0.inputs.set(x)

        self.circuit = circuit

    def save_to_disk(self):
        """
        Saves current model parameter values to disk
        """
        self.circuit.save_to_json(self.exp_dir, self.model_name) ## save current parameter arrays

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        with self.circuit:
            #self.circuit.load_from_dir(self.exp_dir + "/{}".format(self.model_name))
            self.circuit.load_from_dir(model_directory)
            ## note: redo scanner and anything using decorators

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
        out = self.circuit.process(jnp.array([[self.dt*i,self.dt]
                                   for i in range(self.T)]))
        if self.wNorm > 0.:
            self.circuit.norm()
        # self.reset()
        # self.z0.inputs.set(obs)
        # z1e_s = self.circuit.process(jnp.array([[self.dt*i,self.dt] for i in range(self.T)]))
        # self.W1.weights.set(normalize_matrix(self.W1.weights.value, 78.4, order=1, axis=0))
        return z1e_s
