from ngclearn import Context, MethodProcess, JointProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, ScorePatchedSynapse

## Main HiddenPCN model object
class HiddenPCN():
    """
    Structure for constructing the hidden predictive coding network (HiddenPCN) in:

    Whittington, James CR, and Rafal Bogacz. "An approximation of the error
    backpropagation algorithm in a predictive coding network with local hebbian
    synaptic plasticity." Neural computation 29.5 (2017): 1229-1262.

    | Node Name Structure:
    | z0 -(W1)-> e1, z1 -(W2)-> e2, z2 -(W3)-> e3;
    | e2 -(W2.T)-> z1 <- e1, e3 -(W3.T)-> z2 <- e2
    | Note: W1, W2, W3 -> fixed synapses with Hebbian-adapted popup scores

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
    def __init__(
            self, 
            dkey, 
            in_dim=1, 
            out_dim=1, 
            hid1_dim=128, 
            hid2_dim=64, 
            T=10,
            dt=1., 
            tau_m=10., 
            act_fx="tanh", 
            eta=0.001, 
            exp_dir="exp",
            model_name="pc_disc",
            k=0.5,
            weight_init=None,
            loadDir=None, 
            **kwargs
    ):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        ## hard-coded meta-parameters for this model
        optim_type = "adam"

        if loadDir is not None:
            ## build from disk
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                self.z0 = RateCell("z0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.z1 = RateCell(
                    "z1", 
                    n_units=hid1_dim, 
                    tau_m=tau_m, 
                    act_fx=act_fx, 
                    prior=("gaussian", 0.),
                    integration_type="euler"
                )
                self.e1 = ErrorCell("e1", n_units=hid1_dim)
                self.z2 = RateCell(
                    "z2", 
                    n_units=hid2_dim, 
                    tau_m=tau_m, 
                    act_fx=act_fx, 
                    prior=("gaussian", 0.),
                    integration_type="euler"
                )
                self.e2 = ErrorCell("e2", n_units=hid2_dim)
                self.z3 = RateCell("z3", n_units=out_dim, tau_m=0., act_fx="identity")
                self.e3 = ErrorCell("e3", n_units=out_dim)
                ### set up generative/forward synapses
                self.W1 = ScorePatchedSynapse(
                    "W1", 
                    shape=(in_dim, hid1_dim), 
                    eta=eta, k=k,
                    weight_init=weight_init,
                    optim_type=optim_type,
                    sign_value=-1., 
                    key=subkeys[4]
                )
                self.W2 = ScorePatchedSynapse(
                    "W2", 
                    shape=(hid1_dim, hid2_dim), 
                    eta=eta, k=k,
                    weight_init=weight_init,
                    optim_type=optim_type,
                    sign_value=-1., 
                    key=subkeys[5]
                )
                self.W3 = ScorePatchedSynapse(
                    "W3", 
                    shape=(hid2_dim, out_dim), 
                    eta=eta, k=k,
                    weight_init=weight_init,
                    optim_type=optim_type,
                    sign_value=-1., 
                    key=subkeys[6]
                )

                ## wire z0 to e1.mu via W1
                self.z0.zF >> self.W1.inputs
                self.W1.outputs >> self.e1.mu 
                self.z1.z >> self.e1.target 
                ## wire z1 to e2.mu via W2
                self.z1.zF >> self.W2.inputs
                self.W2.outputs >> self.e2.mu 
                self.z2.z >> self.e2.target 
                ## wire z2 to e3.mu via W3
                self.z2.zF >> self.W3.inputs 
                self.W3.outputs >> self.e3.mu 
                self.z3.z >> self.e3.target
                ## wire e2 to z1 via W2.T and e1 to z1 via d/dz1
                self.e2.dmu >> self.W2.post_in
                self.W2.pre_out >> self.z1.j
                self.e1.dtarget >> self.z1.j_td 
                ## wire e3 to z2 via W3.T and e2 to z2 via d/dz2
                self.e3.dmu >> self.W3.post_in
                self.W3.pre_out >> self.z2.j
                self.e2.dtarget >> self.z2.j_td
                ## wire e3 to z3 via d/dz3
                #self.z3.j_td << self.e3.dtarget

                ## setup W1 for its 2-factor Hebbian update
                self.z0.zF >> self.W1.pre 
                self.e1.dmu >> self.W1.post
                ## setup W2 for its 2-factor Hebbian update
                self.z1.zF >> self.W2.pre
                self.e2.dmu >> self.W2.post
                ## setup W3 for its 2-factor Hebbian update
                self.z2.zF >> self.W3.pre
                self.e3.dmu >> self.W3.post

                ## construct inference / projection model
                self.q0 = RateCell("q0", n_units=in_dim, tau_m=0., act_fx="identity")
                self.q1 = RateCell("q1", n_units=hid1_dim, tau_m=0., act_fx=act_fx)
                self.q2 = RateCell("q2", n_units=hid2_dim, tau_m=0., act_fx=act_fx)
                self.q3 = RateCell("q3", n_units=out_dim, tau_m=0., act_fx="identity")
                self.eq3 = ErrorCell("eq3", n_units=out_dim)

                ## wire q0 -(W1)-> q1, q1 -(W2)-> q2, q2 -(W3)-> q3
                self.q0.zF >> self.W1.project_input
                self.W1.project_output >> self.q1.j
                self.q1.zF >> self.W2.project_input
                self.W2.project_output >> self.q2.j
                self.q2.zF >> self.W3.project_input
                self.W3.project_output >> self.q3.j
                #self.eq3.mu = self.q3.z
                ## wire q3 to qe3
                #self.q3.z >> self.eq3.target

                advance_process = (MethodProcess(name="advance_process")
                                   >> self.z0.advance_state
                                   >> self.z1.advance_state
                                   >> self.z2.advance_state
                                   >> self.z3.advance_state
                                   >> self.W1.advance_state
                                   >> self.W2.advance_state
                                   >> self.W3.advance_state
                                   >> self.e1.advance_state
                                   >> self.e2.advance_state
                                   >> self.e3.advance_state)

                reset_process = (MethodProcess(name="reset_process")
                                 >> self.q0.reset
                                 >> self.q1.reset
                                 >> self.q2.reset
                                 >> self.q3.reset
                                 >> self.eq3.reset
                                 >> self.z0.reset
                                 >> self.z1.reset
                                 >> self.z2.reset
                                 >> self.z3.reset
                                 >> self.e1.reset
                                 >> self.e2.reset
                                 >> self.e3.reset)

                evolve_process = (MethodProcess(name="evolve_process")
                                  >> self.W1.evolve
                                  >> self.W2.evolve
                                  >> self.W3.evolve)

                project_process = (MethodProcess(name="project_process")
                                   >> self.q0.advance_state
                                   >> self.W1.advance_state
                                   >> self.q1.advance_state
                                   >> self.W2.advance_state
                                   >> self.q2.advance_state
                                   >> self.W3.advance_state
                                   >> self.q3.advance_state
                                   >> self.eq3.advance_state)

                self.reset = reset_process
                self.advance = advance_process
                self.evolve = evolve_process
                self.project = project_process

    def clamp_input(self, x):
        self.z0.j.set(x)
        self.q0.j.set(x)

    def clamp_target(self, y):
        self.z3.j.set(y)

    def clamp_infer_target(self, y):
        self.eq3.target.set(y)

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.W1.save(model_dir)
            self.W2.save(model_dir)
            self.W3.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            #self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        processes = self.circuit.get_objects_by_type("process") ## obtain all saved processes within this context
        self.advance = processes.get("advance_process")
        self.reset = processes.get("reset_process")
        self.evolve = processes.get("evolve_process")
        self.project = processes.get("project_process")

        nodes = self.circuit.get_components("q0", "q1", "q2", "q3", "eq3",
                                            "z0", "z1", "z2", "z3",
                                            "e1", "e2", "e3",
                                            "W1", "W2", "W3")
        (self.q0, self.q1, self.q2, self.q3, self.eq3, self.z0, self.z1, self.z2,
         self.z3, self.e1, self.e2, self.e3, self.W1, self.W2, self.W3) = nodes


    def process(self, obs, lab, adapt_synapses=True):
        ## can think of the HiddenPCN as doing "PEM" -- projection, expectation, then maximization
        ## w.r.t. the popup scores (the M-step re-selects the sub-network; synaptic weights stay fixed)
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        self.reset.run()

        ## Perform P-step (projection step)
        self.clamp_input(obs)
        self.clamp_infer_target(_lab)
        self.project.run(t=0., dt=1.)

        ## initialize dynamics of generative model latents to projected states
        self.z1.z.set(self.q1.z.get())
        self.z2.z.set(self.q2.z.get())
        # ### Note: e1 = 0, e2 = 0 at initial conditions
        self.e3.dmu.set(self.eq3.dmu.get())
        self.e3.dtarget.set(self.eq3.dtarget.get())
        ## get projected prediction (from the P-step)
        y_mu_inf = self.q3.z.get()

        ## skip E/M steps if just doing test-time inference (we just project)
        EFE = 0.            ## expected free energy
        y_mu = 0.
        if adapt_synapses:
            ## Perform several E-steps
            for ts in range(0, self.T):
                self.clamp_input(obs)
                self.clamp_target(_lab)
                self.advance.run(t=ts, dt=1.)
            ## get settled prediction
            y_mu = self.e3.mu.get()

            ## calculate approximate EFE
            L1 = self.e1.L.get()
            L2 = self.e2.L.get()
            L3 = self.e3.L.get()
            EFE = L3 + L2 + L1           ## expected free energy

            ##  Perform (optional) M-step (scheduled score updates)
            if adapt_synapses == True:
                self.evolve.run(t=self.T, dt=1.)

        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        return self.q2.z.get()

    def _get_norm_string(self): ## debugging routine
        _W1 = self.W1.weights.get()
        _W2 = self.W2.weights.get()
        _W3 = self.W3.weights.get()
        _S1 = self.W1.scores.get()
        _S2 = self.W2.scores.get()
        _S3 = self.W3.scores.get()
        _norms = "W1: {} W2: {} W3: {}\n S1: {} S2: {} S3: {}".format(jnp.linalg.norm(_W1),
                                                                      jnp.linalg.norm(_W2),
                                                                      jnp.linalg.norm(_W3),
                                                                      jnp.linalg.norm(_S1),
                                                                      jnp.linalg.norm(_S2),
                                                                      jnp.linalg.norm(_S3))
        return _norms

