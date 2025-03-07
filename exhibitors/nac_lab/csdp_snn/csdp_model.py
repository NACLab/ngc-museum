from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit, nn
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.operations import summation
from ngclearn.components import (VarTrace, BernoulliCell, SLIFCell, RateCell,
                                 StaticSynapse, HebbianSynapse)
from custom.CSDPSynapse import CSDPSynapse
from custom.goodnessModCell import GoodnessModCell
from custom.maskedErrorCell import MaskedErrorCell as ErrorCell
from ngclearn.utils.model_utils import softmax
from img_utils import csdp_deform #vrotate, rand_rotate,
#from ngclearn.utils.model_utils import normalize_matrix
import ngclearn.utils.weight_distribution as dist

def reset_synapse(syn, batch_size, synapse_type="hebb"):
    pad = jnp.zeros((batch_size, syn.shape[0])) ## input side
    syn.inputs.set(pad)
    pad = jnp.zeros((batch_size, syn.shape[1]))  ## output side
    syn.outputs.set(pad)
    ## reset statistic compartments
    if synapse_type == "hebb":
        syn.pre.set(pad)
        syn.post.set(pad)
    elif synapse_type == "csdp":
        syn.preSpike.set(pad)
        syn.postSpike.set(pad)
        syn.preTrace.set(pad)
        syn.postTrace.set(pad)

def reset_bernoulli(bern, batch_size):
    pad = jnp.zeros((batch_size, bern.n_units))
    bern.inputs.set(pad)
    bern.outputs.set(pad)
    bern.tols.set(pad)

def reset_errcell(ecell, batch_size):
    pad = jnp.zeros((batch_size, ecell.n_units))
    ecell.mu.set(pad)
    ecell.dmu.set(pad)
    ecell.target.set(pad)
    ecell.dtarget.set(pad)
    ecell.modulator.set(pad + 1.)
    ecell.mask.set(pad + 1.)

def reset_goodnesscell(gcell, batch_size):
    pad = jnp.zeros((batch_size, gcell.n_units))
    gcell.inputs.set(pad)
    gcell.modulator.set(pad + 1.)
    gcell.contrastLabels.set(jnp.zeros((batch_size, 1)))

def reset_ratecell(rcell, batch_size):
    pad = jnp.zeros((batch_size, rcell.n_units))
    rcell.j.set(pad)
    rcell.j_td.set(pad)
    rcell.z.set(pad)
    rcell.zF.set(pad)

def reset_lif(lif, batch_size):
    pad = jnp.zeros((batch_size, lif.n_units))
    lif.j.set(pad)
    lif.v.set(pad)
    lif.s.set(pad)
    lif.tols.set(pad)
    lif.rfr.set(pad + lif.refract_T)
    lif.surrogate.set(pad + 1.)

def reset_trace(trace, batch_size):
    pad = jnp.zeros((batch_size, trace.n_units))
    trace.outputs.set(pad)
    trace.inputs.set(pad)
    trace.trace.set(pad)

class CSDP_SNN():

    def __init__(self, dkey, in_dim=1, out_dim=1, hid_dim=1024, hid_dim2=1024,
                 batch_size=1, eta=0.002, T=40, dt=3., learn_recon=False,
                 algo_type="supervised", exp_dir="exp", model_name="snn_csdp",
                 load_model_dir=None, load_param_subdir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        if load_model_dir is None:
            makedir(exp_dir)
            makedir(exp_dir + "/filters")
            makedir(exp_dir + "/raster")

        dkey, *subkeys = random.split(dkey, 20)
        self.T = T  ## num discrete time steps to simulate
        self.dt = dt  ## integration time constant

        ## hard-coded model meta-parameters
        self.algo_type = algo_type
        self.learn_recon = learn_recon
        # spiking cell parameters
        tau_m = 100.  # ms ## membrane time constant (paper used 2 ms)
        vThr = 0.055  # 0.8 ## membrane potential threshold (to emit a spike)
        R_m = 0.1  # 1. ## input resistance (to sLIF cells)
        inh_R = 0.01 #* 6  # 0.035 ## inhibitory resistance
        rho_b = 0.001  ## adaptive threshold sparsity constant
        tau_tr = 13. # 23 or 5 worked less well

        # Synaptic initialization conditions
        weightInit = dist.uniform(amin=-1., amax=1.) #("uniform", -1., 1.)
        biasInit = None # dist.constant(value=0.)  # ("constant", 0., 0.)

        # CSDP-specific meta-parameters
        self.use_rot = False  ## for unsupervised csdp only
        self.alpha = 0.5  ## for unsupervised csdp only
        optim_type = "adam" 
        goodnessThr1 = goodnessThr2 = 10. 
        use_dyn_threshold = False #True
        nonneg_w = False  ## should non-lateral synapses be constrained to be positive-only?

        eta_w = eta
        if batch_size >= 200:  ## heuristic learning problem scaling
            eta_w = 0.002
            w_decay = 0.00005
        elif batch_size >= 100:
            eta_w = 0.001
            w_decay = 0.00006
        elif batch_size >= 50:
            eta_w = 0.001
            w_decay = 0.00007
        elif batch_size >= 20:
            eta_w = 0.00075
            w_decay = 0.00008
        elif batch_size >= 10:
            eta_w = 0.00055
            w_decay = 0.00009
        else:
            eta_w = 0.0004
            w_decay = 0.0001
        soft_bound = False

        if algo_type == "unsupervised":
            goodnessThr1 = goodnessThr2 = 10. #7. # 10. #9.
            use_dyn_threshold = False
            soft_bound = False
            self.use_rot = False ## set to True to use full but slower negative sample generator
        ## else, supervised setting as above

        ################################################################################

        if load_model_dir is not None:
            ## build from disk
            self.load_from_disk(load_model_dir, load_param_subdir)
        else:
            batch_size = 1  # batch_size = batch_size * 2
            with Context("Circuit") as self.circuit:
                self.z0 = BernoulliCell("z0", n_units=in_dim,
                                        batch_size=batch_size, key=subkeys[0])
                self.W1 = CSDPSynapse(
                    name="W1", shape=(in_dim, hid_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=nonneg_w, w_decay=w_decay, resist_scale=R_m,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[1]
                )
                self.z1 = SLIFCell( ## layer 1
                    name="z1", n_units=hid_dim, tau_m=tau_m, resist_m=1., #resist_m=R_m
                    thr=vThr, resist_inh=0., refract_time=0., thr_gain=0.,
                    thr_leak=0., rho_b=rho_b, sticky_spikes=False,
                    thr_jitter=0.025, batch_size=batch_size, key=subkeys[2]
                )
                self.W2 = CSDPSynapse(
                    name="W2", shape=(hid_dim, hid_dim2), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=nonneg_w, w_decay=w_decay, resist_scale=R_m,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[3]
                )
                self.z2 = SLIFCell( ## layer 2
                    name="z2", n_units=hid_dim2, tau_m=tau_m, resist_m=1., #resist_m=R_m
                    thr=vThr, resist_inh=0.,  refract_time=0., thr_gain=0.,
                    thr_leak=0., rho_b=rho_b, sticky_spikes=False,
                    thr_jitter=0.025, batch_size=batch_size, key=subkeys[4]
                )
                self.V2 = CSDPSynapse( ## top-down recurrent synapses from z2 to z1
                    name="V2", shape=(hid_dim2, hid_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=nonneg_w, w_decay=w_decay, resist_scale=R_m,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[5]
                )
                ## set up lateral synapses
                self.M1 = CSDPSynapse( ## top-down recurrent syn from z2 to z1
                    name="M1", shape=(hid_dim, hid_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=True, w_decay=w_decay, resist_scale=inh_R,
                    is_hollow=True, w_sign=-1.,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[6]
                )
                self.M2 = CSDPSynapse(  ## top-down recurrent syn from z2 to z1
                    name="M2", shape=(hid_dim2, hid_dim2), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=True, w_decay=w_decay, resist_scale=inh_R,
                    is_hollow=True, w_sign=-1.,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[7]
                )
                self.zy = SLIFCell(
                    name="zy", n_units=out_dim, tau_m=tau_m, resist_m=1.,
                    thr=vThr, resist_inh=0., # resist_m=R_m
                    refract_time=0., thr_gain=0., thr_leak=0., rho_b=rho_b,
                    sticky_spikes=False, thr_jitter=0.025,
                    batch_size=batch_size, key=subkeys[8]
                )  ## layer 3 - classification layer
                self.ey = ErrorCell(name="ey", n_units=out_dim)
                self.C2 = HebbianSynapse(
                    name="C2", shape=(hid_dim, out_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, resist_scale=1., #resist_scale=R_m
                    w_bound=1., w_decay=0., sign_value=-1., optim_type=optim_type,
                    pre_wght=1., post_wght=R_m, is_nonnegative=nonneg_w,
                    key=subkeys[9]
                )
                self.C3 = HebbianSynapse(
                    name="C3", shape=(hid_dim2, out_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, resist_scale=1., #resist_scale=R_m
                    w_bound=1., w_decay=0., sign_value=-1., optim_type=optim_type,
                    pre_wght=1., post_wght=R_m, is_nonnegative=nonneg_w,
                    key=subkeys[10]
                )
                if self.learn_recon:
                    self.zR = SLIFCell(
                        name="zR", n_units=in_dim, tau_m=tau_m, resist_m=1., #resist_m=R_m
                        thr=vThr, resist_inh=0.,
                        refract_time=0., thr_gain=0., thr_leak=0., rho_b=0.,
                        sticky_spikes=False, thr_jitter=0.025,
                        batch_size=batch_size, key=subkeys[11]
                    )  ## reconstruction layer
                    self.eR = ErrorCell(name="eR", n_units=in_dim)
                    self.R1 = HebbianSynapse(
                        name="R1", shape=(hid_dim, in_dim), eta=eta_w,
                        weight_init=weightInit, bias_init=biasInit,
                        resist_scale=1., w_bound=1., w_decay=0., sign_value=-1., # resist_scale=R_m
                        optim_type=optim_type, pre_wght=1., post_wght=R_m,
                        is_nonnegative=nonneg_w, key=subkeys[12]
                    )
                ### context/class units (only used if model is supervised)
                self.z3 = BernoulliCell(
                    "z3", n_units=out_dim, batch_size=batch_size, key=subkeys[13]
                )
                if self.algo_type == "supervised":
                    self.V3y = CSDPSynapse(
                        name="V3y", shape=(out_dim, hid_dim2),
                        eta=eta_w,
                        weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                        is_nonnegative=nonneg_w, w_decay=w_decay,
                        resist_scale=R_m,
                        optim_type=optim_type, soft_bound=soft_bound,
                        key=subkeys[13]
                    )
                    self.V2y = CSDPSynapse(
                        name="V2y", shape=(out_dim, hid_dim), eta=eta_w,
                        weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                        is_nonnegative=nonneg_w, w_decay=w_decay,
                        resist_scale=R_m, # * skipcontext_syn_factor, 
                        optim_type=optim_type,
                        soft_bound=soft_bound, key=subkeys[14]
                    ) # no ablation applied here since context is an "input" at least to hidden layer

                ## create goodness modulators (1 per layer)
                self.g1 = GoodnessModCell(name="g1", n_units=hid_dim, threshold=goodnessThr1,
                                          use_dyn_threshold=use_dyn_threshold)
                self.g2 = GoodnessModCell(name="g2", n_units=hid_dim2, threshold=goodnessThr2,
                                          use_dyn_threshold=use_dyn_threshold)
                # no g3 since layer 3 is a possible classification layer
                ########################################################################
                ## static (state) recordings of spike values at t - dt
                self.z0_prev = RateCell(
                    name="z0_prev", n_units=in_dim, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                self.z1_prev = RateCell(
                    name="z1_prev", n_units=hid_dim, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                self.z2_prev = RateCell(
                    name="z2_prev", n_units=hid_dim2, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                self.z3_prev = RateCell(
                    name="z3_prev", n_units=out_dim, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                ########################################################################
                ### add trace variables
                self.tr0 = VarTrace(
                    "tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="lin",
                    batch_size=batch_size, a_delta=0., key=subkeys[15]
                )
                self.tr1 = VarTrace(
                    "tr1", n_units=hid_dim, tau_tr=tau_tr, decay_type="lin",
                    a_delta=0., batch_size=batch_size, key=subkeys[15]
                )
                self.tr2 = VarTrace(
                    "tr2", n_units=hid_dim2, tau_tr=tau_tr, decay_type="lin",
                    a_delta=0., batch_size=batch_size, key=subkeys[15]
                )
                self.tr3 = VarTrace(
                    "tr3", n_units=out_dim, tau_tr=tau_tr, decay_type="lin",
                    a_delta=0., batch_size=batch_size, key=subkeys[15]
                )

                ## wire nodes to their respective traces
                self.tr1.inputs << self.z1.s
                self.tr2.inputs << self.z2.s
                self.tr3.inputs << self.zy.s
                ## wire traces of nodes to their respective goodness modulators
                self.g1.inputs << self.tr1.trace
                self.g2.inputs << self.tr2.trace
                ## wires nodes to their respective previous time-step recordings
                self.z0_prev.j << self.z0.outputs
                self.z1_prev.j << self.z1.s
                self.z2_prev.j << self.z2.s
                self.z3_prev.j << self.z3.outputs
                ########################################################################

                ### Wire layers together - z1 and z2 to zy classifier
                self.ey.mu << self.tr3.outputs #self.z3.outputs
                self.ey.target << self.z3.outputs #self.tr3.outputs
                self.C2.inputs << self.z1.s
                self.C3.inputs << self.z2.s
                self.zy.j << summation(self.C2.outputs, self.C3.outputs)

                if self.learn_recon:
                    self.R1.inputs << self.z1.s
                    self.zR.j << self.R1.outputs
                    self.tr0.inputs << self.zR.s
                    self.eR.mu << self.zR.s #self.tr0.trace
                    self.eR.target << self.z0.outputs
                    ## set up R1 plasticity rule
                    self.R1.pre << self.z1.s
                    self.R1.post << self.eR.dmu

                ## layer 0 to 1
                self.W1.inputs << self.z0.outputs
                self.M1.inputs << self.z1_prev.zF ## z1 lateral
                self.V2.inputs << self.z2_prev.zF ## z2-to-z1
                if self.algo_type == "supervised":
                    self.V2y.inputs << self.z3_prev.zF ## context to z2
                    self.z1.j << summation(self.W1.outputs, self.M1.outputs,
                                           self.V2.outputs, self.V2y.outputs)
                else:
                    self.z1.j << summation(self.W1.outputs, self.M1.outputs,
                                           self.V2.outputs)
                ## layer 1 to 2
                self.W2.inputs << self.z1.s
                self.M2.inputs << self.z2_prev.zF  ## z1 lateral
                if self.algo_type == "supervised":
                    self.V3y.inputs << self.z3_prev.zF  ## context to z2
                    self.z2.j << summation(self.W2.outputs, self.M2.outputs,
                                           self.V3y.outputs)
                else:
                    self.z2.j << summation(self.W2.outputs, self.M2.outputs)

                ## wire relevant compartment stats to trigger plasticity rules
                self.C2.pre << self.z1_prev.zF
                self.C2.post << self.ey.dmu
                self.C3.pre << self.z2_prev.zF
                self.C3.post << self.ey.dmu
                ## update to W1
                self.W1.preSpike << self.z0_prev.zF
                self.W1.postSpike << self.z1.s
                self.W1.preTrace << self.z0_prev.zF
                self.W1.postTrace << self.g1.modulator
                ## update to M1
                self.M1.preSpike << self.z1_prev.zF
                self.M1.postSpike << self.z1.s
                self.M1.preTrace << self.z1_prev.zF
                self.M1.postTrace << self.g1.modulator
                ## update to W2
                self.W2.preSpike << self.z1_prev.zF
                self.W2.postSpike << self.z2.s
                self.W2.postTrace << self.g2.modulator
                ## update to M2
                self.M2.preSpike << self.z2_prev.zF
                self.M2.postSpike << self.z2.s
                self.M2.preTrace << self.z2_prev.zF
                self.M2.postTrace << self.g2.modulator
                ## update to V2
                self.V2.preSpike << self.z2_prev.zF
                self.V2.postSpike << self.z1.s
                self.V2.preTrace << self.z2_prev.zF
                self.V2.postTrace << self.g1.modulator
                if self.algo_type == "supervised": ## updates to Y3 (V3y) and Y2 (V2y)
                    self.V3y.preSpike << self.z3_prev.zF
                    self.V3y.postSpike << self.z2.s
                    self.V3y.preTrace << self.z3_prev.zF
                    self.V3y.postTrace << self.g2.modulator
                    self.V2y.preSpike << self.z3_prev.zF
                    self.V2y.postSpike << self.z1.s
                    self.V2y.preTrace << self.z3_prev.zF
                    self.V2y.postTrace << self.g1.modulator
                ########################################################################
                ## make key commands known to model
                if self.algo_type == "supervised":
                    exec_path = [self.z0_prev, self.z1_prev, self.z2_prev, self.z3_prev,
                                 self.W1, self.V2, self.W2, self.V3y, self.V2y,
                                 self.M1, self.M2, self.C2, self.C3,
                                 self.z0, self.z1, self.z2, self.z3, self.zy,
                                 self.tr1, self.tr2, self.tr3,
                                 self.g1, self.g2, self.ey]
                    evolve_path = [self.W1, self.V2, self.W2, self.V2y,
                                   self.V3y, self.M1, self.M2, self.C2, self.C3]
                    save_path = [self.W1, self.V2, self.W2, self.V2y, self.V3y,
                                 self.M1, self.M2, self.C2, self.C3,
                                 self.z1, self.z2, self.zy]
                else:  ## unsupervised
                    exec_path = [self.z0_prev, self.z1_prev, self.z2_prev, self.z3_prev,
                                 self.W1, self.V2, self.W2, self.M1,
                                 self.M2, self.C2, self.C3,
                                 self.z0, self.z1, self.z2, self.z3, self.zy,
                                 self.tr1, self.tr2, self.tr3,
                                 self.g1, self.g2, self.ey]
                    evolve_path = [self.W1, self.V2, self.W2,
                                   self.M1, self.M2, self.C2, self.C3]
                    save_path = [self.W1, self.V2, self.W2, self.M1, self.M2,
                                 self.C2, self.C3, self.z1, self.z2, self.zy]
                if self.learn_recon:
                    recon_path = [self.R1, self.zR, self.tr0, self.eR]
                    exec_path = exec_path + recon_path
                    evolve_path = evolve_path + [self.R1]
                    save_path = save_path + [self.zR, self.R1]

                reset_cmd, reset_args = self.circuit.compile_by_key(
                    *exec_path, compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_by_key(
                    *exec_path, compile_key="advance_state")
                evolve_cmd, evolve_args = self.circuit.compile_by_key(
                    *evolve_path, compile_key="evolve")
                self.dynamic()
        ## some book-keeping to ease component retrieval
        self.traces = ["tr0", "tr1", "tr2", "tr3"]
        self.ecells = ["ey"]
        self.gcells = ["g1", "g2"]
        self.input_cells = ["z0", "z3"]
        self.lifs = ["z1", "z2", "zy"]
        self.ratecells = ["z0_prev", "z1_prev", "z2_prev", "z3_prev"]
        if self.algo_type == "supervised":
            self.csdp_synapses = ["W1", "W2", "V2", "V2y", "V3y", "M1", "M2"]
        else:
            self.csdp_synapses = ["W1", "W2", "V2", "M1", "M2"]
        self.hebb_synapses = ["C2", "C3"]
        self.ecells = []
        if self.learn_recon:
            self.ecells = self.ecells + ["eR"]
            self.lifs = self.lifs + ["zR"]
            self.hebb_synapses = self.hebb_synapses + ["R1"]
        self.saveable_comps = (self.input_cells + self.lifs + self.ratecells +
                               self.ecells + self.gcells + self.traces +
                               self.hebb_synapses + self.csdp_synapses)

    def dynamic(self): ## create dynamic commands for circuit
        z0, z1, z2, z3, zy = self.circuit.get_components("z0", "z1", "z2", "z3", "zy")
        g1, g2, ey, eR = self.circuit.get_components("g1", "g2", "ey", "eR")

        self.circuit.add_command(
            wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(
            wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(
            wrap_command(jit(self.circuit.evolve)), name="evolve")

        @Context.dynamicCommand
        def clamp_input(x):
            z0.inputs.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            z3.inputs.set(y)

        @Context.dynamicCommand
        def clamp_mod_labels(labs):
            g1.contrastLabels.set(labs)
            g2.contrastLabels.set(labs)
            ey.mask.set(labs)
            eR.mask.set(labs)

    def save_to_disk(self, save_dir, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/{}".format(self.exp_dir, self.model_name,
                                          save_dir)
            makedir(model_dir)
            for comp_name in self.saveable_comps:
                comp = self.circuit.components.get(comp_name)
                comp.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name) ## save current parameter arrays

    def load_from_disk(self, model_directory, param_subdir="/custom"):
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory, custom_folder=param_subdir)
            ## note: redo scanner and anything using decorators
            self.dynamic()
        self.W1 = self.circuit.components.get("W1")
        self.W2 = self.circuit.components.get("W2")
        self.M1 = self.circuit.components.get("M1")
        self.M2 = self.circuit.components.get("M2")
        self.V2 = self.circuit.components.get("V2")
        self.C2 = self.circuit.components.get("C2")
        self.C3 = self.circuit.components.get("C3")
        self.V2y = self.circuit.components.get("V2y")
        self.V3y = self.circuit.components.get("V3y")
        self.R1 = self.circuit.components.get("R1")
        self.z0_prev = self.circuit.components.get("z0_prev")
        self.z1_prev = self.circuit.components.get("z1_prev")
        self.z2_prev = self.circuit.components.get("z2_prev")
        self.z3_prev = self.circuit.components.get("z3_prev")
        self.z0 = self.circuit.components.get("z0")
        self.z1 = self.circuit.components.get("z1")
        self.z2 = self.circuit.components.get("z2")
        self.z3 = self.circuit.components.get("z3")
        self.zy = self.circuit.components.get("zy")
        self.zR = self.circuit.components.get("zR")
        self.tr0 = self.circuit.components.get("tr0")
        self.tr1 = self.circuit.components.get("tr1")
        self.tr2 = self.circuit.components.get("tr2")

    def get_synapse_stats(self, param_name="W1"):
        _W1 = self.circuit.components.get(param_name).weights.value
        msg = "{}:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
            param_name, jnp.amin(_W1), jnp.amax(_W1), jnp.mean(_W1),
            jnp.linalg.norm(_W1)
        )
        return msg

    def viz_receptive_fields(
            self, param_name, field_shape, fname, transpose_params=False,
            n_fields_to_view=-1
    ):
        _W1 = self.circuit.components.get(param_name).weights.value
        if 0 < n_fields_to_view < _W1.shape[1]:
            _W1 = _W1[:, 0:n_fields_to_view] ## extract a subset of filters to viz
        if transpose_params:
            _W1 = _W1.T
        visualize([_W1], [field_shape], fname)

    def process(self, Xb, Yb, dkey, adapt_synapses=False, collect_spikes=False,
                collect_rate_codes=False, lab_estimator="softmax",
                collect_recon=True, Xb_neg=None, Yb_neg=None):
        dkey, *subkeys = random.split(dkey, 2)
        if adapt_synapses:
            ## create negative sensory samples
            if self.algo_type == "supervised":
                Yb_neg = random.uniform(subkeys[0], Yb.shape, minval=0.,
                                        maxval=1.) * (1. - Yb)
                Yb_neg = nn.one_hot(jnp.argmax(Yb_neg, axis=1),
                                    num_classes=Yb.shape[1], dtype=jnp.float32)
                Xb_neg = Xb
            else:  ## algo_type is unsupervised
                if Xb_neg is None:
                    bsize = Xb.shape[0]
                    _Xb = jnp.expand_dims(jnp.reshape(Xb, (bsize, 28, 28)), axis=3)
                    Xb_neg, Yb_neg = csdp_deform(subkeys[0], _Xb, Yb,
                                                 alpha=self.alpha,
                                                 use_rot=self.use_rot)
                    Xb_neg = jnp.reshape(jnp.squeeze(Xb_neg, axis=3),
                                         (bsize, 28 * 28))
            ## concatenate the samples
            _Xb = jnp.concatenate((Xb, Xb_neg), axis=0)
            _Yb = jnp.concatenate((Yb, Yb_neg), axis=0)
            mod_signal = jnp.concatenate((jnp.ones((Xb.shape[0], 1)),
                                          jnp.zeros((Xb_neg.shape[0], 1))),
                                         axis=0)
        else:
            _Yb = Yb * 0  ## we nix the labels during inference/test-time
            _Xb = Xb
            mod_signal = jnp.ones((Xb.shape[0], 1))

        self.circuit.reset()

        batch_size = Xb.shape[0]
        if adapt_synapses:
            batch_size = batch_size * 2
        for name in self.traces:
            reset_trace(self.circuit.components.get(name), batch_size)
        for name in self.ecells:
            reset_errcell(self.circuit.components.get(name), batch_size)
        for name in self.gcells:
            reset_goodnesscell(self.circuit.components.get(name), batch_size)
        for name in self.input_cells:
            reset_bernoulli(self.circuit.components.get(name), batch_size)
        for name in self.lifs:
            reset_lif(self.circuit.components.get(name), batch_size)
        for name in self.ratecells:
            reset_ratecell(self.circuit.components.get(name), batch_size)
        if adapt_synapses:
            for name in self.hebb_synapses:
                reset_synapse(self.circuit.components.get(name), batch_size,
                              synapse_type="hebb")
            for name in self.csdp_synapses:
                reset_synapse(self.circuit.components.get(name), batch_size,
                              synapse_type="csdp")

        s0_mu = _Xb * 0
        y_count = 0.  # prediction spike train
        self.z3.inputs.set(_Yb) #.outputs
        self.z3_prev.z.set(_Yb)  # .inputs
        T = self.T + 1
        R1 = 0.
        R2 = 0.
        R3 = 0.
        for ts in range(1, T):
            self.circuit.clamp_input(_Xb)
            self.circuit.clamp_target(_Yb)
            self.circuit.clamp_mod_labels(mod_signal)
            self.circuit.advance(t=ts*self.dt, dt=self.dt)
            if adapt_synapses:
                self.circuit.evolve(t=ts * self.dt, dt=self.dt)

            y_count = self.zy.s.value + y_count
            if self.learn_recon or collect_recon:
                s0_mu = self.tr0.outputs.value + s0_mu
            if collect_rate_codes:
                R1 = self.z1.s.value + R1
                R2 = self.z2.s.value + R2

        ## estimate total goodness
        s0_mu = s0_mu / T
        ## estimate output distribution
        if lab_estimator == "softmax":
            y_hat = softmax(y_count)
        else:
            y_hat = y_count
        if collect_rate_codes:
            R1 = R1 / T #self.T # T+1
            R2 = R2 / T #self.T #T+1
            R3 = y_hat

        return y_hat, y_count, R1, R2, R3, s0_mu
