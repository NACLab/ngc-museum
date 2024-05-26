from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.commands import Command
from ngcsimlib.operations import summation
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
import time, sys

from ngclearn.utils.model_utils import softmax
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse

## PCN model co-routines
def load_model(model_dir, exp_dir="exp", model_name="pc_disc", dt=1., T=10):
    _key = random.PRNGKey(time.time_ns())
    ## load circuit from disk
    circuit = Controller()
    circuit.load_from_dir(directory=model_dir)

    model = PCN(_key, in_dim=1, out_dim=1, save_init=False)
    model.circuit = circuit
    model.exp_dir = exp_dir
    model.model_dir = "{}/{}/custom".format(exp_dir, model_name)
    model.dt = dt
    model.T = T
    return model


def tie_compartments(circuit, target, source, compartmentName):
    """
    Ties/shares parameter values from a source (cell) component's compartment with
    another target (cell) component's compartment of the same name.

    Args:
        circuit: controller object to perform tying on

        target: target component to give shallow copy of compartment value to

        source: source component to draw compartment value from

        compartmentName: name of compartment to tie/share values across source
            and target (cell) components
    """
    _value = circuit.components[source].compartments[compartmentName]
    circuit.components[target].compartments[compartmentName] = _value

def tie_parameters(circuit, target, source, transpose_source=False, share_bias=False):
    """
    Ties/shares parameter values from a source component synaptic cable with a target
    component synaptic cable (i.e., gives target a shallow copy of the source's
    parameters).

    Args:
        circuit: controller object to perform tying on

        target: target component to give shallow copy of params to

        source: source component to draw param values from

        transpose_source: should source "weights" parameters be transposed
            before shallow copying (Default: False)

        share_bias: should source "biases" be shared with target (Default: False)
    """
    source_W = circuit.components[source].weights
    if transpose_source == True:
        source_W = source_W.T
    circuit.components[target].weights = source_W
    if share_bias == True:
        circuit.components[target].biases = circuit.components[source].biases

## Main PCN model object

class PCN():
    """
    Structure for constructing the predictive coding network (PCN) in:

    Whittington, James CR, and Rafal Bogacz. "An approximation of the error
    backpropagation algorithm in a predictive coding network with local hebbian
    synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

    | Node Name Structure:
    | z0 -(W1)-> e1, z1 -(W1)-> e2, z2 -(W3)-> e3;
    | e2 -(E2)-> z1 <- e1, e3 -(E3)-> z2 <- e2
    | Note: W1, W2, W3 -> Hebbian-adapted synapses

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        out_dim: output dimensionality

        hid1_dim: dimensionality of 1st layer of internal neuronal cells

        hid2_dim: dimensionality of 2nd layer of internal neuronal cells

        T: number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

        tau_m: membrane time constant of hidden/internal neuronal layers

        act_fx: activation function to use for internal neuronal layers

        exp_dir: experimental directory to save model results

        model_name: unique model name to stamp the output files/dirs with

        save_init: save model at initialization/first configuration time (Default: True)
    """
    def __init__(self, dkey, in_dim, out_dim, hid1_dim=128, hid2_dim=64, T=10,
                 dt=1., tau_m=10., act_fx = "tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        ## hard-coded meta-parameters for this model
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3

        if loadDir is not None:
            ## build from disk
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = RateCell("z0", n_units=in_dim, tau_m=0., act_fx="identity",
                              key=subkeys[0])
                self.z1 = RateCell("z1", n_units=hid1_dim, tau_m=tau_m, act_fx=act_fx,
                              prior=("gaussian",0.), integration_type="euler",
                              key=subkeys[1])
                self.e1 = ErrorCell("e1", n_units=hid1_dim)
                self.z2 = RateCell("z2", n_units=hid2_dim, tau_m=tau_m, act_fx=act_fx,
                              prior=("gaussian", 0.), integration_type="euler",
                              key=subkeys[2])
                self.e2 = ErrorCell("e2", n_units=hid2_dim)
                self.z3 = RateCell("z3", n_units=out_dim, tau_m=0., act_fx="identity",
                              key=subkeys[3])
                self.e3 = ErrorCell("e3", n_units=out_dim)
                ### set up generative/forward synapses
                self.W1 = HebbianSynapse("W1", shape=(in_dim, hid1_dim), eta=eta,
                                         wInit=("uniform", wlb, wub),
                                         bInit=("constant", 0., 0.), w_bound=0.,
                                         optim_type=optim_type, signVal=-1.,
                                         key=subkeys[4])
                self.W2 = HebbianSynapse("W2", shape=(hid1_dim, hid2_dim), eta=eta,
                                         wInit=("uniform", wlb, wub),
                                         bInit=("constant", 0., 0.), w_bound=0.,
                                         optim_type=optim_type, signVal=-1.,
                                         key=subkeys[5])
                self.W3 = HebbianSynapse("W3", shape=(hid2_dim, out_dim), eta=eta,
                                         wInit=("uniform", wlb, wub),
                                         bInit=("constant", 0., 0.), w_bound=0.,
                                         optim_type=optim_type, signVal=-1.,
                                         key=subkeys[6])
                ## set up feedback/error synapses
                self.E2 = HebbianSynapse("E2", shape=(hid2_dim, hid1_dim), eta=0.,
                                         wInit=("uniform", wlb, wub), w_bound=0.,
                                         signVal=-1., key=subkeys[4])
                self.E3 = HebbianSynapse("E3", shape=(out_dim, hid2_dim), eta=0.,
                                         wInit=("uniform", wlb, wub), w_bound=0.,
                                         signVal=-1., key=subkeys[5])

                ## wire z0 to e1.mu via W1
                self.W1.inputs << self.z0.zF
                self.e1.mu << self.W1.outputs
                self.e1.target << self.z1.z
                ## wire z1 to e2.mu via W2
                self.W2.inputs << self.z1.zF
                self.e2.mu << self.W2.outputs
                self.e2.target << self.z2.z
                ## wire z2 to e3.mu via W3
                self.W3.inputs << self.z2.zF
                self.e3.mu << self.W3.outputs
                self.e3.target << self.z3.z
                ## wire e2 to z1 via W2.T and e1 to z1 via d/dz1
                self.E2.inputs << self.e2.dmu
                self.z1.j << self.E2.outputs
                self.z1.j_td << self.e1.dtarget
                ## wire e3 to z2 via W3.T and e2 to z2 via d/dz2
                self.E3.inputs << self.e3.dmu
                self.z2.j << self.E3.outputs
                self.z2.j_td << self.e2.dtarget
                ## wire e3 to z3 via d/dz3
                #self.z3.j_td << self.e3.dtarget

                ## setup W1 for its 2-factor Hebbian update
                self.W1.pre << self.z0.zF
                self.W1.post << self.e1.dmu
                ## setup W2 for its 2-factor Hebbian update
                self.W2.pre << self.z1.zF
                self.W2.post << self.e2.dmu
                ## setup W3 for its 2-factor Hebbian update
                self.W3.pre << self.z2.zF
                self.W3.post << self.e3.dmu

                ## construct inference / projection model
                self.q0 = RateCell("q0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.q1 = RateCell("q1", n_units=hid1_dim, tau_m=0., act_fx=act_fx)
                self.q2 = RateCell("q2", n_units=hid2_dim, tau_m=0., act_fx=act_fx)
                self.q3 = RateCell("q3", n_units=out_dim, tau_m=0., act_fx="identity")
                self.eq3 = ErrorCell("eq3", n_units=out_dim)
                self.Q1 = HebbianSynapse("Q1", shape=(in_dim, hid1_dim),
                                         bInit=("constant", 0., 0.), key=subkeys[0])
                self.Q2 = HebbianSynapse("Q2", shape=(hid1_dim, hid2_dim),
                                         bInit=("constant", 0., 0.), key=subkeys[0])
                self.Q3 = HebbianSynapse("Q3", shape=(hid2_dim, out_dim),
                                         bInit=("constant", 0., 0.), key=subkeys[0])
                ## wire q0 -(Q1)-> q1, q1 -(Q2)-> q2, q2 -(Q3)-> q3
                self.Q1.inputs << self.q0.zF
                self.q1.j << self.Q1.outputs
                self.Q2.inputs << self.q1.zF
                self.q2.j << self.Q2.outputs
                self.Q3.inputs << self.q2.zF
                self.q3.j << self.Q3.outputs
                #self.eq3.mu = self.q3.z
                ## wire q3 to qe3
                self.eq3.target << self.q3.z

                reset_cmd, reset_args = self.circuit.compile_command_key(
                                                self.q0, self.q1, self.q2, self.q3, self.eq3,
                                                self.z0, self.z1, self.z2, self.z3,
                                                self.e1, self.e2, self.e3,
                                            compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_command_key(
                                                    self.E2, self.E3,
                                                    self.z0, self.z1, self.z2, self.z3,
                                                    self.W1, self.W2, self.W3,
                                                    self.e1, self.e2, self.e3,
                                                compile_key="advance_state") ## E-step
                evolve_cmd, evolve_args = self.circuit.compile_command_key(
                                                    self.W1, self.W2, self.W3,
                                                compile_key="evolve") ## M-step
                project_cmd, project_args = self.circuit.compile_command_key(
                                                    self.q0, self.Q1, self.q1, self.Q2,
                                                    self.q2, self.Q3, self.q3, self.eq3,
                                                compile_key="advance_state", name="project") ## project
                self.dynamic()

        # ## save JSON structure to disk once
        # if save_init == True:
        #     model.save_to_json(directory="exp", model_name=model_name)
        # self.model_dir = "{}/{}/custom".format(exp_dir, model_name)
        # if save_init == True:
        #     model.save(dir=self.model_dir) ## save current parameter arrays
        # self.circuit = model # embed model construct to agent "circuit"

    def dynamic(self):## create dynamic commands for circuit
        #from ngcsimlib.utils import get_current_context
        #context = get_current_context()
        vars = self.circuit.get_components("q0", "q1", "q2", "q3", "eq3",
                                           "Q1", "Q2", "Q3",
                                           "z0", "z1", "z2", "z3",
                                           "e1", "e2", "e3",
                                           "W1", "W2", "W3", "E2", "E3")
        q0, q1, q2, q3, eq3, Q1, Q2, Q3, z0, z1, z2, z3, e1, e2, e3, W1, W2, W3, E2, E3 = vars
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.eq3 = eq3
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3
        self.z0 = z0
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.E2 = E2
        self.E3 = E3
        self.nodes = [q0, q1, q2, q3, eq3, Q1, Q2, Q3, z0, z1, z2, z3, e1, e2, e3, W1, W2, W3, E2, E3]

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")
        self.circuit.add_command(wrap_command(jit(self.circuit.project)), name="project")

        @Context.dynamicCommand
        def clamp_input(x):
            z0.j.set(x)
            q0.j.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            z3.j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            eq3.target.set(y)

        # @scanner
        # def process(compartment_values, args):
        #     t = args[0]
        #     dt = args[1]
        #     compartment_values = self.circuit.advance_state(compartment_values, t, dt)
        #     compartment_values = self.circuit.evolve(compartment_values, t, dt)
        #     return compartment_values, (compartment_values[self.z1.zF.path],
        #                                 compartment_values[self.z2.zF.path],
        #                                 compartment_values[self.z3.zF.path])

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only == True:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
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

    def process(self, obs, lab, adapt_synapses=True):
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        #self.circuit.reset(do_reset=True)
        self.circuit.reset()

        ## pin/tie inference synapses to be exactly equal to the forward ones
        self.Q1.weights.set(self.W1.weights.value)
        self.Q1.biases.set(self.W1.biases.value)
        self.Q2.weights.set(self.W2.weights.value)
        self.Q2.biases.set(self.W2.biases.value)
        self.Q3.weights.set(self.W3.weights.value)
        self.Q3.biases.set(self.Q3.biases.value)
        ## pin/tie feedback synapses to transpose of forward ones
        self.E2.weights.set(jnp.transpose(self.W2.weights.value))
        self.E3.weights.set(jnp.transpose(self.W3.weights.value))
        # tie_parameters(self.circuit, "Q1", "W1", transpose_source=False, share_bias=True)
        # tie_parameters(self.circuit, "Q2", "W2", transpose_source=False, share_bias=True)
        # tie_parameters(self.circuit, "Q3", "W3", transpose_source=False, share_bias=True)
        # ## pin/tie feedback synapses to transpose of forward ones
        # tie_parameters(self.circuit, "E2", "W2", transpose_source=True, share_bias=False)
        # tie_parameters(self.circuit, "E3", "W3", transpose_source=True, share_bias=False)

        ## Perform P-step (projection step)
        # self.circuit.clamp_input(x=obs) ## clamp to q0 & z0 input compartments
        # self.circuit.clamp_infer_target(target=_lab)
        # self.circuit.project(t=0, dt=0.) ## do projection/inference
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        self.circuit.project(0., 1.) ## do projection/inference

        ## initialize dynamics of generative model latents to projected states
        # tie_compartments(self.circuit, "z1", "q1", compartmentName="z")
        # tie_compartments(self.circuit, "z2", "q2", compartmentName="z")
        # ###transfer_compartments(self.circuit, "z3", "q3", compartmentName="z")
        self.z1.z.set(self.q1.z.value)
        self.z2.z.set(self.q2.z.value)
        # ### Note: e1 = 0, e2 = 0 at initial conditions
        # tie_compartments(self.circuit, "e3", "eq3", compartmentName="dmu")
        # tie_compartments(self.circuit, "e3", "eq3", compartmentName="dtarget")
        self.e3.dmu.set(self.eq3.dmu.value)
        self.e3.dtarget.set(self.eq3.dtarget.value)

        #y_mu_inf = self.circuit.components["q3"].compartments["z"] ## get projected prediction
        y_mu_inf = self.q3.z.value ## get projected prediction

        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses == True:
            ## Perform several E-steps
            for ts in range(0, self.T):
                #print("###################### {} #########################".format(ts))
                # self.circuit.clamp_input(x=obs) ## clamp data to z0 & q0 input compartments
                # self.circuit.clamp_target(target=_lab) ## clamp data to e3.target
                # self.circuit.runCycle(t=ts*self.dt, dt=self.dt)
                self.circuit.clamp_input(obs) ## clamp data to z0 & q0 input compartments
                self.circuit.clamp_target(_lab) ## clamp data to e3.target
                self.circuit.advance(ts, 1.)

            #y_mu = self.circuit.components["e3"].compartments["mu"] ## get settled prediction
            y_mu = self.e3.mu.value ## get settled prediction

            # L1 = self.circuit.components["e1"].compartments["L"]
            # L2 = self.circuit.components["e2"].compartments["L"]
            # L3 = self.circuit.components["e3"].compartments["L"]
            L1 = self.e1.L.value
            L2 = self.e2.L.value
            L3 = self.e3.L.value

            EFE = L3 + L2 + L1
            ## Perform (optional) M-step (scheduled synaptic updates)
            if adapt_synapses == True:
                #self.circuit.evolve(t=self.T, dt=self.dt)
                self.circuit.evolve(ts, 1.)
            print(y_mu_inf)
            print("...")
            print(y_mu)
            sys.exit(0)
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        #return self.circuit.components["q2"].compartments["z"]
        return self.q2.z.value

    def _get_norm_string(self): ## debugging routine
        _W1 = self.W1.weights.value
        _W2 = self.W2.weights.value
        _W3 = self.W3.weights.value
        _b1 = self.W1.biases.value
        _b2 = self.W2.biases.value
        _b3 = self.W3.biases.value
        _norms = "W1: {} W2: {} W3: {}\n b1: {} b2: {} b3: {}".format(jnp.linalg.norm(_W1),
                                                                      jnp.linalg.norm(_W2),
                                                                      jnp.linalg.norm(_W3),
                                                                      jnp.linalg.norm(_b1),
                                                                      jnp.linalg.norm(_b2),
                                                                      jnp.linalg.norm(_b3))
        return _norms

    # def _get_norm_string(self): ## debugging routine
    #     _W1 = self.circuit.components.get("W1").weights
    #     _W2 = self.circuit.components.get("W2").weights
    #     _W3 = self.circuit.components.get("W3").weights
    #     _b1 = self.circuit.components.get("W1").biases
    #     _b2 = self.circuit.components.get("W2").biases
    #     _b3 = self.circuit.components.get("W3").biases
    #     _norms = "W1: {} W2: {} W3: {}\n b1: {} b2: {} b3: {}".format(jnp.linalg.norm(_W1),
    #                                                                 jnp.linalg.norm(_W2),
    #                                                                 jnp.linalg.norm(_W3),
    #                                                                 jnp.linalg.norm(_b1),
    #                                                                 jnp.linalg.norm(_b2),
    #                                                                 jnp.linalg.norm(_b3))
    #     return _norms
