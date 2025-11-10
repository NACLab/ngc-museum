from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess, JointProcess
from ngclearn.components.input_encoders.poissonCell import PoissonCell
from ngclearn.components.neurons.spiking.LIFCell import LIFCell
from ngclearn.components.synapses.staticSynapse import StaticSynapse
from ngclearn.components.synapses.hebbian.traceSTDPSynapse import TraceSTDPSynapse
from ngclearn.components.other.varTrace import VarTrace
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.operations import Summation

from ngcsimlib.global_state import stateManager

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
                self.z0 = PoissonCell("z0", n_units=in_dim, target_freq=63.75, key=subkeys[0])
                self.W1 = TraceSTDPSynapse(
                    "W1", shape=(in_dim, hid_dim), A_plus=Aplus, A_minus=Aminus, eta=1., pretrace_target=0.,
                    weight_init=DistributionGenerator.uniform(0.0, 0.3), key=subkeys[1]
                )
                self.z1e = LIFCell(
                    "z1e", n_units=hid_dim, tau_m=tau_m_e, resist_m=tau_m_e / dt, thr=-52., v_rest=-65.,
                    v_reset=-60., tau_theta=1e7, theta_plus=0.05, refract_time=5., one_spike=True, key=subkeys[2]
                )
                self.z1i = LIFCell(
                    "z1i", n_units=hid_dim, tau_m=tau_m_i, resist_m=tau_m_i / dt, thr=-40., v_rest=-60.,
                    v_reset=-45., tau_theta=0., refract_time=5., one_spike=False, key=subkeys[3]
                )

                # ie -> inhibitory to excitatory; ei -> excitatory to inhibitory
                #       (eta = 0 means no learning)
                self.W1ie = StaticSynapse(
                    "W1ie", shape=(hid_dim, hid_dim),
                                          weight_init=DistributionGenerator.constant(-120., hollow=True),
                                          key=subkeys[4])
                self.W1ei = StaticSynapse(
                    "W1ei", shape=(hid_dim, hid_dim),
                    weight_init=DistributionGenerator.constant(22.5, eye=True), key=subkeys[5]
                )
                self.tr0 = VarTrace("tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="exp", a_delta=0., key=subkeys[6])
                self.tr1 = VarTrace("tr1", n_units=hid_dim, tau_tr=tau_tr, decay_type="exp", a_delta=0., key=subkeys[7])

                self.z0.outputs >> self.W1.inputs
                self.z1i.s >> self.W1ie.inputs
                Summation(self.W1.outputs, self.W1ie.outputs) >> self.z1e.j

                self.z1e.s >> self.W1ei.inputs
                self.W1ei.outputs >> self.z1i.j

                self.z0.outputs >> self.tr0.inputs
                self.z1e.s >> self.tr1.inputs

                self.tr0.trace >> self.W1.preTrace
                self.z0.outputs >> self.W1.preSpike
                self.tr1.trace >> self.W1.postTrace
                self.z1e.s >> self.W1.postSpike

                self.advance_proc = (MethodProcess(name="advance_process")
                                     >> self.W1.advance_state
                                     >> self.W1ie.advance_state
                                     >> self.W1ei.advance_state
                                     >> self.z0.advance_state
                                     >> self.z1e.advance_state
                                     >> self.z1i.advance_state
                                     >> self.tr0.advance_state
                                     >> self.tr1.advance_state)

                self.reset_proc = (MethodProcess(name="reset_process")
                                   >> self.z0.reset
                                   >> self.z1e.reset
                                   >> self.z1i.reset
                                   >> self.tr0.reset
                                   >> self.tr1.reset
                                   >> self.W1.reset
                                   >> self.W1ie.reset
                                   >> self.W1ei.reset)

                self.evolve_proc = (MethodProcess(name="evolve_process")
                                    >> self.W1.evolve)

                self.forward_proc = (JointProcess(name="forward_process")
                                     >> self.advance_proc
                                     >> self.evolve_proc)

                self.advance_proc.watch(self.z1e.s, self.z1e.v) #, self.z1e.j)

    def norm(self):
        self.W1.weights.set(normalize_matrix(self.W1.weights.get(), self.wNorm, order=1, axis=0))

    def clamp(self, x):
        self.z0.inputs.set(x)

    def get_synapse_stats(self):
        """
        Print basic statistics of W1 to string

        Returns:
            string containing min, max, mean, and L2 norm of W1
        """
        _W1 = self.W1.weights.get()
        msg = "W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
            jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1), jnp.linalg.norm(_W1)
        )
        return msg

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only: ## this condition allows to only write actual parameter values w/in components to disk
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.z1e.save(model_dir)
        else: ## this saves the whole model form (JSON structure as well as parameter values)
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            #self.circuit.save_to_json(self.exp_dir, self.model_name) ## save current parameter arrays

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        #with Context("Circuit") as self.circuit:
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        #self.advance_proc = self.circuit.get_objects("advance_process", objectType="process")
        processes = self.circuit.get_objects_by_type("process") ## obtain all saved processes within this context
        self.advance_proc = processes.get("advance_process")
        self.reset_proc = processes.get("reset_process")
        self.evolve_proc = processes.get("evolve_process")
        self.forward_proc = processes.get("forward_process")

        W1, z0, z1e = self.circuit.get_components("W1", "z0", "z1e")
        self.W1 = W1
        self.z0 = z0
        self.z1e = z1e

    def viz_receptive_fields(self, fname, field_shape):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            fname: plot fname name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot
        """
        _W1 = self.W1.weights.get()
        visualize([_W1], [field_shape], self.exp_dir + "/filters/{}".format(fname))

    def process(self, obs, adapt_synapses=True, collect_spike_train=False):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T. Note that the observed pattern will be converted
        to a Poisson spike train with maximum frequency of 63.75 Hertz.

        Note that this model assumes batch sizes of one (online learning).

        Args:
            obs: observed pattern to input to this spiking neural model

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
        #z0, z1e, z1i, W1 = self.circuit.get_components("z0", "z1e", "z1i", "W1")

        inputs = jnp.array(self.forward_proc.pack_rows(self.T, t=lambda x: x, dt=self.dt))
        #print(inputs.shape)

        self.reset_proc.run()
        self.clamp(obs)
        if adapt_synapses:
            stateManager.state, outputs = self.forward_proc.scan(inputs)
            self.norm()
        else:
            stateManager.state, outputs = self.advance_proc.scan(inputs)
            # print(self.z0.inputs.get())
            # print(len(outputs))
            # print(outputs[1][0, :])
            # print(jnp.abs(outputs[1][0, :] - outputs[1][11, :]))
            # exit()

        spike_out = outputs[0]
        return spike_out
