from ngcsimlib.controller import Controller
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random
import time

class DC_SNN_Model():

    def __init__(self, dkey, in_dim, hid1_dim=128, hid2_dim=64, T=200, dt=0.1, exp_dir="exp",
                 model_name="snn_stdp", **kwargs):
        self.exp_dir = exp_dir
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        #makedir(exp_dir + "/raster")

        dkey, *subkeys = random.split(dkey, 10)

        optim_type = "adam"
        eta = 0.002

        ## set up model with layers of neuronal cells
        model = Controller()
        z0 = model.add_component("rate", name="z0", n_units=in_dim, tau_m=0.,
                                 act_fx="identity", leakRate=0., key=subkeys[0])
        e0 = model.add_component("error", name="e0", n_units=in_dim)
        z1 = model.add_component("rate", name="z1", n_units=hid1_dim, tau_m=tau_m,
                                 act_fx="tanh", leakRate=0., key=subkeys[1])
        e1 = model.add_component("error", name="e1", n_units=hid1_dim)
        z2 = model.add_component("rate", name="z2", n_units=hid2_dim, tau_m=tau_m,
                                 act_fx="tanh", leakRate=0., key=subkeys[2])
        z3 = model.add_component("rate", name="z3", n_units=in_dim, tau_m=0.,
                                 act_fx="identity", leakRate=0., key=subkeys[3])
        ### set up generative/forward synapses
        W1 = model.add_component("hebbian", name="W1", shape=(in_dim, hid1_dim),
                                 eta=eta, wInit=("uniform", -0.3, 0.3), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[4])
        W2 = model.add_component("hebbian", name="W2", shape=(hid1_dim, hid2_dim),
                                 eta=eta, wInit=("uniform", -0.3, 0.3), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[5])
        W3 = model.add_component("hebbian", name="W3", shape=(hid2_dim, out_dim),
                                 eta=eta, wInit=("uniform", -0.3, 0.3), w_bound=0.,
                                 optim_type=optim_type, signVal=-1., key=subkeys[6])
        ## set up feedback/error synapses
        E1 = model.add_component("hebbian", name="E2", shape=(hid2_dim, hid1_dim),
                                 eta=0., wInit=("uniform", -0.3, 0.3), w_bound=0.,
                                 signVal=-1., key=subkeys[4])
        E3 = model.add_component("hebbian", name="E3", shape=(out_dim, hid2_dim),
                                 eta=0., wInit=("uniform", -0.3, 0.3), w_bound=0.,
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

        ## checks that everything is valid within model structure
        #model.verify_cycle()

        ## make key commands known to model
        ## will need to clamp to z3 and e3.target = x
        model.add_command("reset", command_name="reset",
                          component_names=[z0.name, z1.name, z2.name, z3.name,
                                           e1.name, e2.name, e3.name],
                          reset_name="do_reset")
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
                          component_names=[z0.name], compartment=z0.inputCompartmentName(),
                          clamp_name="x")
        model.add_command("clamp", command_name="clamp_target",
                          component_names=[e0.name], compartment=e0.targetName(),
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

        ## TODO: port over CSDP option to plug in optimizer into Hebb synapse

    def process(self, obs, lab, adapt_synapses=True):
        ## run analytical model
        #W1 = model.components["W1"].weights
        #W2 = model.components["W2"].weights
        #z0_mu, z1_mu, z1_f, z2_f, e0, e1, dW1, dW2 = calc(Xb, W1, W2, K=T, beta=dt/tau_m)
        # print("+++++++++++++++++ Ref ++++++++++++++++++++++")
        # print("z2: ",z2_f)
        # print("mu1: ",z1_mu)
        # print("e1: ",e1)
        # print("z1: ",z1_f)
        # print("mu0: ",z0_mu)
        # print("e0: ",e0)
        # print("++++++++++++++++++++++++++++++++++++++++++++")

        self.circuit.reset(do_reset=True)
        ## pin feedback synapses to transpose of forward ones
        self.circuit.components["E2"].weights = (self.circuit.components["W2"].weights).T
        self.circuit.components["E3"].weights = (self.circuit.components["W3"].weights).T
        ## Perform E-step
        for ts in range(0, T):
            #print("###################### {} #########################".format(ts))
            self.circuit.clamp_input(x=obs) #x=inp) ## clamp data to z0.input
            self.circuit.clamp_target(target=lab) ## clamp data to e3.target
            #print("e0.comp = ",model.components["e0"].compartments)
            self.circuit.runCycle(t=ts*dt, dt=dt)
        ## Perform (optional) M-step (scheduled synaptic updates)
        if adapt_synapses == True:
            self.circuit.evolve(t=T, dt=dt)
