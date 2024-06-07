#from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
#from jax.lax import scan
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.operations import summation
from ngclearn.components.other.varTrace import VarTrace
from ngclearn.components.input_encoders.poissonCell import PoissonCell
from ngclearn.components.neurons.spiking.LIFCell import LIFCell
from ngclearn.components.synapses import TraceSTDPSynapse, StaticSynapse
from ngclearn.utils.model_utils import normalize_matrix
import ngclearn.utils.weight_distribution as dist

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

        loadDir: directory to load model from, overrides initialization/model
            object creation if non-None (Default: None)
    """
    # Define Functions
    def __init__(self, dkey, in_dim=1, hid_dim=100, T=200, dt=1., exp_dir="exp",
                 model_name="snn_stdp", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        #makedir("{}/{}".format(exp_dir, model_name))
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

        if loadDir is not None:
            ## build from disk
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = PoissonCell("z0", n_units=in_dim, max_freq=63.75, key=subkeys[0])
                self.W1 = TraceSTDPSynapse("W1", shape=(in_dim, hid_dim),
                                           A_plus=Aplus, A_minus=Aminus, eta=1.,
                                           pretrace_target=0.,
                                           weight_init=dist.uniform(0.0, 0.3),
                                           key=subkeys[1])
                self.z1e = LIFCell("z1e", n_units=hid_dim, tau_m=tau_m_e,
                                   resist_m=tau_m_e/dt, thr=-52., v_rest=-65.,
                                   v_reset=-60., tau_theta=1e7, theta_plus=0.05,
                                   refract_time=5., one_spike=True, key=subkeys[2])
                self.z1i = LIFCell("z1i", n_units=hid_dim, tau_m=tau_m_i,
                                   resist_m=tau_m_i/dt, thr=-40., v_rest=-60.,
                                   v_reset=-45., tau_theta=0., refract_time=5.,
                                   one_spike=False, key=subkeys[3])

                # ie -> inhibitory to excitatory; ei -> excitatory to inhibitory
                #       (eta = 0 means no learning)
                self.W1ie = StaticSynapse("W1ie", shape=(hid_dim, hid_dim),
                                          weight_init=dist.hollow(-120.),
                                          key=subkeys[4])
                self.W1ei = StaticSynapse("W1ei", shape=(hid_dim, hid_dim),
                                          weight_init=dist.eye(22.5),
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

                reset_cmd, reset_args = self.circuit.compile_by_key(
                                            self.z0, self.z1e, self.z1i,
                                            self.tr0, self.tr1,
                                            self.W1, self.W1ie, self.W1ei,
                                            compile_key="reset")

                advance_cmd, advance_args = self.circuit.compile_by_key(
                                                self.W1, self.W1ie, self.W1ei,
                                                self.z0, self.z1e, self.z1i,
                                                self.tr0, self.tr1,
                                                compile_key="advance_state")
                evolve_cmd, evolve_args = self.circuit.compile_by_key(self.W1, compile_key="evolve")


                #self.circuit.add_command(wrap_command(jit(reset_cmd)), name="reset")
                self.dynamic()

    def dynamic(self):## create dynamic commands for circuit
        #from ngcsimlib.utils import get_current_context
        #context = get_current_context()
        W1, z0, z1e = self.circuit.get_components("W1", "z0", "z1e")
        self.W1 = W1
        self.z0 = z0
        self.z1e = z1e

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")

        @Context.dynamicCommand
        def norm():
            W1.weights.set(normalize_matrix(W1.weights.value, self.wNorm, order=1, axis=0))

        @Context.dynamicCommand
        def clamp(x):
            z0.inputs.set(x)

        @scanner
        def process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
            compartment_values = self.circuit.evolve(compartment_values, t=_t, dt=_dt)
            return compartment_values, compartment_values[self.z1e.s.path]


    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only == True:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.z1e.save(model_dir)
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
            #self.circuit.load_from_dir(self.exp_dir + "/{}".format(self.model_name))
            self.circuit.load_from_dir(model_directory)
            ## note: redo scanner and anything using decorators
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
        out = self.circuit.process(jnp.array([[self.dt*i,self.dt]
                                   for i in range(self.T)]))
        if self.wNorm > 0.:
            self.circuit.norm()
        # self.reset()
        # self.z0.inputs.set(obs)
        # out = self.circuit.process(jnp.array([[self.dt*i,self.dt] for i in range(self.T)]))
        # self.W1.weights.set(normalize_matrix(self.W1.weights.value, 78.4, order=1, axis=0))
        return out
