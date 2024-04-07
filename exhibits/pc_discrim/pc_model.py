from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random
import time, sys

class PCN():
    """
    Structure for constructing the predictive coding (network) in:

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
    """
    def __init__(self, dkey, in_dim, out_dim, hid1_dim=128, hid2_dim=64, T=10,
                 dt=1., tau_m=10., act_fx = "tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", **kwargs):
        self.exp_dir = exp_dir
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        #makedir(exp_dir + "/raster")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        #tau_m = 10.
        #act_fx = "tanh"
        optim_type = "adam"
        #eta = 0.002
        wlb = -0.3
        wub = 0.3

        ## set up model with layers of neuronal cells
        model = Controller()
        ## construct core generative model
        z0 = model.add_component("rate", name="z0", n_units=in_dim, tau_m=0.,
                                 act_fx="identity", key=subkeys[0])
        z1 = model.add_component("rate", name="z1", n_units=hid1_dim, tau_m=tau_m,
                                 act_fx=act_fx, prior=("gaussian",0.),
                                 integration_type="euler", key=subkeys[1])
        e1 = model.add_component("error", name="e1", n_units=hid1_dim)
        z2 = model.add_component("rate", name="z2", n_units=hid2_dim, tau_m=tau_m,
                                 act_fx=act_fx, prior=("gaussian", 0.),
                                 integration_type="euler", key=subkeys[2])
        e2 = model.add_component("error", name="e2", n_units=hid2_dim)
        z3 = model.add_component("rate", name="z3", n_units=out_dim, tau_m=0.,
                                 act_fx="identity", key=subkeys[3])
        e3 = model.add_component("error", name="e3", n_units=out_dim)
        ### set up generative/forward synapses
        W1 = model.add_component("hebbian", name="W1", shape=(in_dim, hid1_dim),
                                 eta=eta, wInit=("uniform", wlb, wub), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[4])
        W2 = model.add_component("hebbian", name="W2", shape=(hid1_dim, hid2_dim),
                                 eta=eta, wInit=("uniform", wlb, wub), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[5])
        W3 = model.add_component("hebbian", name="W3", shape=(hid2_dim, out_dim),
                                 eta=eta, wInit=("uniform", wlb, wub), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[6])
        ## set up feedback/error synapses
        E2 = model.add_component("hebbian", name="E2", shape=(hid2_dim, hid1_dim),
                                 eta=0., wInit=("uniform", wlb, wub), w_bound=0.,
                                 signVal=-1., key=subkeys[4])
        E3 = model.add_component("hebbian", name="E3", shape=(out_dim, hid2_dim),
                                 eta=0., wInit=("uniform", wlb, wub), w_bound=0.,
                                 signVal=-1., key=subkeys[5])
        ## wire z0 to e1.mu via W1
        model.connect(z0.name, z0.outputCompartmentName(), W1.name, W1.inputCompartmentName())
        model.connect(W1.name, W1.outputCompartmentName(), e1.name, e1.meanName())
        model.connect(z1.name, z1.rateActivityName(), e1.name, e1.targetName())
        ## wire z1 to e2.mu via W2
        model.connect(z1.name, z1.outputCompartmentName(), W2.name, W2.inputCompartmentName())
        model.connect(W2.name, W2.outputCompartmentName(), e2.name, e2.meanName())
        model.connect(z2.name, z2.rateActivityName(), e2.name, e2.targetName())
        ## wire z2 to e3.mu via W3
        model.connect(z2.name, z2.outputCompartmentName(), W3.name, W3.inputCompartmentName())
        model.connect(W3.name, W3.outputCompartmentName(), e3.name, e3.meanName())
        model.connect(z3.name, z3.rateActivityName(), e3.name, e3.targetName())

        ## wire e2 to z1 via W2.T and e1 to z1 via d/dz1
        model.connect(e2.name, e2.derivMeanName(), E2.name, E2.inputCompartmentName())
        model.connect(E2.name, E2.outputCompartmentName(), z1.name, z1.inputCompartmentName())
        model.connect(e1.name, e1.derivTargetName(), z1.name, z1.pressureName())
        ## wire e3 to z2 via W3.T and e2 to z2 via d/dz2
        model.connect(e3.name, e3.derivMeanName(), E3.name, E3.inputCompartmentName())
        model.connect(E3.name, E3.outputCompartmentName(), z2.name, z2.inputCompartmentName())
        model.connect(e2.name, e2.derivTargetName(), z2.name, z2.pressureName())
        ## wire e3 to z3 via d/dz3
        #model.connect(e3.name, e3.derivTargetName(), z3.name, z3.inputCompartmentName())

        ## setup W1 for its 2-factor Hebbian update
        model.connect(z0.name, z0.outputCompartmentName(), W1.name, W1.presynapticCompartmentName())
        model.connect(e1.name, e1.derivMeanName(), W1.name, W1.postsynapticCompartmentName())
        ## setup W2 for its 2-factor Hebbian update
        model.connect(z1.name, z1.outputCompartmentName(), W2.name, W2.presynapticCompartmentName())
        model.connect(e2.name, e2.derivMeanName(), W2.name, W2.postsynapticCompartmentName())
        ## setup W3 for its 2-factor Hebbian update
        model.connect(z2.name, z2.outputCompartmentName(), W3.name, W3.presynapticCompartmentName())
        model.connect(e3.name, e3.derivMeanName(), W3.name, W3.postsynapticCompartmentName())

        ## construct inference / projection model
        q0 = model.add_component("rate", name="q0", n_units=in_dim, tau_m=0., act_fx="identity")
        q1 = model.add_component("rate", name="q1", n_units=hid1_dim, tau_m=0., act_fx=act_fx)
        q2 = model.add_component("rate", name="q2", n_units=hid2_dim, tau_m=0., act_fx=act_fx)
        q3 = model.add_component("rate", name="q3", n_units=out_dim, tau_m=0., act_fx="identity")
        eq3 = model.add_component("error", name="eq3", n_units=out_dim)
        Q1 = model.add_component("hebbian", name="Q1", shape=(in_dim, hid1_dim), key=subkeys[0])
        Q2 = model.add_component("hebbian", name="Q2", shape=(hid1_dim, hid2_dim), key=subkeys[0])
        Q3 = model.add_component("hebbian", name="Q3", shape=(hid2_dim, out_dim), key=subkeys[0])
        ## wire q0 -(Q1)-> q1, q1 -(Q2)-> q2, q2 -(Q3)-> q3
        model.connect(q0.name, q0.outputCompartmentName(), Q1.name, Q1.inputCompartmentName())
        model.connect(Q1.name, Q1.outputCompartmentName(), q1.name, q1.inputCompartmentName())
        model.connect(q1.name, q1.outputCompartmentName(), Q2.name, Q2.inputCompartmentName())
        model.connect(Q2.name, Q2.outputCompartmentName(), q2.name, q2.inputCompartmentName())
        model.connect(q2.name, q2.outputCompartmentName(), Q3.name, Q3.inputCompartmentName())
        model.connect(Q3.name, Q3.outputCompartmentName(), q3.name, q3.inputCompartmentName())
        ## wire q3 to qe3
        model.connect(q3.name, q3.rateActivityName(), eq3.name, eq3.targetName())

        ## checks that everything is valid within model structure
        #model.verify_cycle()

        ## make key commands known to model
        ## will need to clamp to z3 and e3.target = x
        model.add_command("reset", command_name="reset",
                          component_names=[q0.name, q1.name, q2.name, q3.name, eq3.name,
                                           z0.name, z1.name, z2.name, z3.name,
                                           e1.name, e2.name, e3.name],
                          reset_name="do_reset")
        model.add_command(
            "advance", command_name="project",
            component_names=[q0.name, Q1.name, q1.name, Q2.name,
                             q2.name, Q3.name, q3.name, eq3.name,
                            ]
        )
        model.add_command(
            "advance", command_name="advance",
            component_names=[E2.name, E3.name,
                             z0.name, z1.name, z2.name, z3.name,
                             W1.name, W2.name, W3.name,
                             e1.name, e2.name, e3.name
                            ]
        )
        model.add_command("evolve", command_name="evolve",
                          component_names=[W1.name, W2.name, W3.name])
        model.add_command("clamp", command_name="clamp_input",
                          component_names=[z0.name, q0.name],
                          compartment=z0.inputCompartmentName(),
                          clamp_name="x")
        model.add_command("clamp", command_name="clamp_target",
                          component_names=[z3.name], compartment=z3.inputCompartmentName(),
                          clamp_name="target")
        model.add_command("clamp", command_name="clamp_infer_target",
                          component_names=[eq3.name], compartment=eq3.targetName(),
                          clamp_name="target")
        model.add_command("save", command_name="save",
                          component_names=[W1.name, W2.name, W3.name],
                          directory_flag="dir")

        ## tell model the order in which to run automatic commands
        #model.add_step("clamp_input")
        model.add_step("advance")

        ## save JSON structure to disk once
        model.save_to_json(directory="exp", model_name=model_name)
        self.model_dir = "{}/{}/custom".format(exp_dir, model_name)
        model.save(dir=self.model_dir) ## save current parameter arrays
        self.circuit = model # embed model construct to agent "circuit"

    def viz_receptive_fields(self, fname, field_shape, show_stats=True):
        """
        Generates and saves a plot of the receptive fields for the current state
        of the model's synaptic efficacy values in W1.

        Args:
            fname: plot fname name (appended to end of experimental directory)

            field_shape: 2-tuple specifying expected shape of receptive fields to plot

            show_stats: display to I/O basic statistics of W1
        """
        _W1 = self.circuit.components.get("W1").weights
        visualize([_W1], [field_shape], self.exp_dir + "/filters/{}".format(fname))
        if show_stats == True:
            msg = "W1:\n  min {} \n  max {} \n  mu {} \n  norm {}"
            print(msg.format(jnp.amin(_W1),
                             jnp.amax(_W1),
                             jnp.mean(_W1),
                             jnp.linalg.norm(_W1)))

    def print_norms(self):
        _W1 = self.circuit.components.get("W1").weights
        _W2 = self.circuit.components.get("W2").weights
        _W3 = self.circuit.components.get("W3").weights
        _b1 = self.circuit.components.get("b1").weights
        _b2 = self.circuit.components.get("b2").weights
        _b3 = self.circuit.components.get("b3").weights
        _norms = "W1: {} W2: {} W3: {} b1: {} b2: {} b3: {}".format(jnp.linalg.norm(_W1),
                                                                    jnp.linalg.norm(_W2),
                                                                    jnp.linalg.norm(_W3),
                                                                    jnp.linalg.norm(_b1),
                                                                    jnp.linalg.norm(_b2),
                                                                    jnp.linalg.norm(_b3))
        print(_norms)

    def process(self, obs, lab, adapt_synapses=True):
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        self.circuit.reset(do_reset=True)
        ## pin inference synapses to be exactly equal to the forward ones
        self.circuit.components["Q1"].weights = (self.circuit.components["W1"].weights)
        self.circuit.components["Q2"].weights = (self.circuit.components["W2"].weights)
        self.circuit.components["Q3"].weights = (self.circuit.components["W3"].weights)
        ## pin feedback synapses to transpose of forward ones
        self.circuit.components["E2"].weights = (self.circuit.components["W2"].weights).T
        self.circuit.components["E3"].weights = (self.circuit.components["W3"].weights).T

        ## Perform P-step (projection step)
        self.circuit.clamp_input(x=obs) ## clamp to q0 & z0 input compartments
        self.circuit.clamp_infer_target(target=_lab)
        self.circuit.project(t=0, dt=0.) ## do projection/inference
        ## initialize dynamics of generative model latents to projected states
        self.circuit.components["z1"].compartments["z"] = self.circuit.components["q1"].compartments["z"]
        self.circuit.components["z2"].compartments["z"] = self.circuit.components["q2"].compartments["z"]
        ###self.circuit.components["z3"].compartments["z"] = self.circuit.components["q3"].compartments["z"]
        ## pin projection statistics to main inference components
        ### Note: e1 = 0, e2 = 0 at initial conditions
        self.circuit.components["e3"].compartments["dmu"] = self.circuit.components["eq3"].compartments["dmu"]
        self.circuit.components["e3"].compartments["dtarget"] = self.circuit.components["eq3"].compartments["dtarget"]

        y_mu_inf = self.circuit.components["q3"].compartments["z"] ## get projected prediction
        #print("mu: ",self.circuit.components["q3"].compartments["z"])
        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses == True:
            ## Perform E-step
            for ts in range(0, self.T):
                #print("###################### {} #########################".format(ts))
                self.circuit.clamp_input(x=obs) ## clamp data to z0 & q0 input compartments
                self.circuit.clamp_target(target=_lab) ## clamp data to e3.target
                #print("e0.comp = ",model.components["e0"].compartments)
                self.circuit.runCycle(t=ts*self.dt, dt=self.dt)
            #print("mu: ",self.circuit.components["e3"].compartments["mu"])
            #print(" y: ",self.circuit.components["e3"].compartments["target"])
            #print("---")
            #print(self.circuit.components["z2"].compartments["z"]
            y_mu = self.circuit.components["e3"].compartments["mu"] ## get settled prediction

            L1 = self.circuit.components["e1"].compartments["L"]
            L2 = self.circuit.components["e2"].compartments["L"]
            L3 = self.circuit.components["e3"].compartments["L"]
            EFE = L3 + L2 + L1
            ## Perform (optional) M-step (scheduled synaptic updates)
            if adapt_synapses == True:
                self.circuit.evolve(t=self.T, dt=self.dt)
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE
