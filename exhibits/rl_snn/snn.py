from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess, JointProcess
from ngclearn.components.input_encoders.poissonCell import PoissonCell
from custom import NoisyLIFCell as LIFCell ## Use a modified LIF component for this exhibit
from ngclearn.components.synapses.staticSynapse import StaticSynapse
from ngclearn.components.synapses.modulated.MSTDPETSynapse import MSTDPETSynapse 
from ngclearn.components.other.varTrace import VarTrace
from ngclearn.utils.model_utils import normalize_matrix
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.operations import Summation

from ngcsimlib.global_state import stateManager

from ngclearn.utils.model_utils import normalize_matrix, softmax, one_hot

def init_sparse_syn(dkey, shape, n_minus, n_plus, is_unique=False):
    ## synapse initialization scheme
    nrows, ncols = shape
    dkey, *subkeys = random.split(dkey, nrows + 1)
    empty_syn = jnp.zeros((1, ncols))
    W = []
    ptrs = random.permutation(subkeys[0], ncols)
    s_ptr = 0
    e_ptr = n_minus + n_plus
    for r in range(nrows):
        syn_r = None
        if not is_unique:
            ptrs = random.permutation(subkeys[r], ncols)
            exc = ptrs[0:n_plus]  ##
            inh = ptrs[n_plus:n_plus + n_minus]
            syn_r = empty_syn.at[0, exc].set(1.)
            syn_r = syn_r.at[0, inh].set(-1.)
        else:
            #print(f"{s_ptr}  {e_ptr}")
            exc = ptrs[s_ptr:s_ptr+n_plus]  ##
            inh = ptrs[s_ptr+n_plus:e_ptr]
            s_ptr += (n_minus + n_plus)
            e_ptr = (s_ptr + (n_minus + n_plus))
            syn_r = empty_syn.at[0, exc].set(1.)
            syn_r = syn_r.at[0, inh].set(-1.)
            #print(syn_r)
        W.append(syn_r)
    W = jnp.concatenate(W, axis=0)
    return W

class SNN():
    """
    Structure for constructing a spiking neural controller for a simple operant conditioning task.

    Args:
        dkey: JAX seeding key

        in_dim: input dimensionality

        out_dim: output dimensionality, i.e, number of control states or discrete actions

        n_hid: dimensionality of the representation layer of neuronal cells

        T: length of stimulus window, i.e., number of discrete time steps to simulate neuronal dynamics

        dt: integration time constant

    """
    def __init__(
            self, dkey, in_dim, out_dim, n_hid, T=100, dt=1., **kwargs):

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        n_minus = 3 ## number of inhibitory synapses in encoding layer (for fixed feature layer)
        n_plus = 6 ## number of excitatory synapses in encoding layer (for fixed feature layer)
        p_minus = 1. ## trace constant that a spike clamps to (for eligibility traces)
        tau_w2 = 2e7 ## decay coefficient for control layer (rstdp)
        Aplus2 = 1. ## LTP for control layer
        Aminus2 = 0.01 ## LTD for control layer
        x_tar2 = 0.0
        nu2 = 0.01 * 10
        tau_m_e = 100.
        tau_theta = 0. #1000.
        theta_plus = 0. #0.05
        tau_p = 20. ## spike trace time constant
        tau_e2 = 40. ## eligibility time constant for control layer
        w_max_init = 0.55 #0.99 # 0.3 #0.25 #0.9 #0.3 # 0.35
        norm_perc = 0.5 #0.5 # 0.5
        self.wnorm1 = norm_perc * in_dim
        self.wnorm2 = norm_perc * n_hid
        self.normalize = True
        inhR = 4. ## cross-inhibitory strength for feature layer
        inhR2 = 4. ## cross-inhibitory strength for control layer
        lif_noise_eps = 0. ## 0.2
        self.reset_volt_thresholds = True

        with Context("Circuit") as self.circuit:
            z0 = PoissonCell("z0", n_units=in_dim, target_freq=127.5, key=subkeys[0])
            ## input-to-hidden synapses
            W1 = StaticSynapse(
                "W1", shape=(in_dim, n_hid), weight_init=DistributionGenerator.uniform(-1., 1.),
                resist_scale=4., key=subkeys[0]
            )
            W1.weights.set(init_sparse_syn(subkeys[0], W1.weights.get().shape, n_minus, n_plus))  # jnp.sign(W1.weights.value))
            V1 = StaticSynapse( ## lateral inhibitory synapses for feature layer
                "V1", shape=(n_hid, n_hid), weight_init=DistributionGenerator.constant(1., hollow=True),
                resist_scale=-inhR, key=subkeys[1]
            )
            #V1.weights.set(V1.weights.value * random.uniform(subkeys[1], shape=(n_hid, n_hid), minval=0.4, maxval=1.))
            z1 = LIFCell( ## projection/hidden layer
                "z1", n_units=n_hid, tau_m=tau_m_e, resist_m=1., v_decay=1., thr=-52., v_rest=-65., v_reset=-65.,
                eps_scale=lif_noise_eps, tau_theta=tau_theta, theta_plus=theta_plus, refract_time=0., enforce_wta=False,
                v_min=None, key=subkeys[2]
            )
            ## hidden-to-place synapses
            V2 = StaticSynapse( ## lateral inhibitory synapses for control layer
                "V2", shape=(out_dim, out_dim), weight_init=DistributionGenerator.constant(1., hollow=True),
                resist_scale=-inhR2, key=subkeys[3]
            )
            #V2.weights.set(V2.weights.value * random.uniform(subkeys[3], shape=(out_dim, out_dim), minval=0.4, maxval=1.))
            W2 = MSTDPETSynapse(
                "W2", shape=(n_hid, out_dim), A_plus=Aplus2, A_minus=Aminus2, tau_elg=tau_e2, eta=nu2, tau_w=tau_w2,
                pretrace_target=x_tar2, weight_init=DistributionGenerator.uniform(0.01, w_max_init), resist_scale=1.,
                w_bound=1., key=subkeys[4]
            )
            z2 = LIFCell(
                "z2", n_units=out_dim, tau_m=tau_m_e, resist_m=1., v_decay=1., thr=-52., v_rest=-65., v_reset=-65.,
                eps_scale=lif_noise_eps, tau_theta=tau_theta, theta_plus=theta_plus, refract_time=0., enforce_wta=False,
                v_min=None, key=subkeys[5]
            )
            ## set up traces
            W2_p1_tr = VarTrace( ## pre-synaptic trace for W2
                "W2_p1_tr", n_units=n_hid, tau_tr=tau_p, decay_type="exp", P_scale=p_minus, a_delta=0., key=subkeys[0]
            )
            W2_p2_tr = VarTrace( ## post-synaptic trace for W2
                "W2_p2_tr", n_units=out_dim, tau_tr=tau_p, decay_type="exp", P_scale=p_minus, a_delta=0., key=subkeys[0]
            )

            ## wire z0 to z1
            z0.outputs >> W1.inputs
            z1.s >> V1.inputs
            Summation(W1.outputs, V1.outputs) >> z1.j
            ## wire z1 to z2
            z1.s >> W2.inputs
            z2.s >> V2.inputs
            Summation(W2.outputs, V2.outputs) >> z2.j
            # wire cells to their respective traces
            z1.s >> W2_p1_tr.inputs ## for W2's plasticity
            z2.s >> W2_p2_tr.inputs ## for W2's plasticity

            # wire relevant compartment statistics to synaptic cables W1 and W2 (to drive learning rules)
            W2_p1_tr.trace >> W2.preTrace
            z1.s >> W2.preSpike
            W2_p2_tr.trace >> W2.postTrace
            z2.s >> W2.postSpike

            ## inference process step
            self.advance_proc = (MethodProcess(name="advance_process")
                                 >> V1.advance_state
                                 >> V2.advance_state 
                                 >> W1.advance_state 
                                 >> W2.advance_state 
                                 >> z0.advance_state 
                                 >> z1.advance_state 
                                 >> z2.advance_state 
                                 >> W2_p1_tr.advance_state 
                                 >> W2_p2_tr.advance_state)
            ## reset-to-baseline-values process step
            self.reset_proc = (MethodProcess(name="reset_process")
                               >> V1.reset
                               >> V2.reset
                               >> W1.reset
                               >> W2.reset
                               >> z0.reset
                               >> z1.reset
                               >> z2.reset
                               >> W2_p1_tr.reset
                               >> W2_p2_tr.reset)
            ## intermediate evolutionary process step
            self.evolve_proc = (MethodProcess(name="evolve_process")
                                >> W2.evolve)
            ## (reward-)modulated evolutionary process step
            self.modulated_evolve_proc = (MethodProcess(name="modulated_evolve")
                                          >> W2.evolve)
            ## compound adaptation process (inference+learning)
            self.adapt_proc = (JointProcess(name="forward_process")
                               >> self.advance_proc
                               >> self.evolve_proc)
            self.advance_proc.watch(z1.v, z1.s, z2.v, z2.s) ## watch these compartments

            self.z0 = z0
            self.z1 = z1
            self.z2 = z2
            self.W1 = W1
            self.V1 = V1
            self.W2 = W2
            self.V2 = V2 

            if self.normalize:
                self.norm()

    def norm(self): ## synapse normalization step
        self.W2.weights.set(normalize_matrix(self.W2.weights.get(), self.wnorm2, order=1, axis=0))

    def clamp(self, x): ## input-stimulus clamping step
        self.z0.inputs.set(x)

    def set_to_resting_state(self):
        """
        Sets all internal states of this model to their resting values.
        """
        self.reset_proc.run()
        z1, z2 = self.circuit.get_components("z1", "z2")
        if self.reset_volt_thresholds:
            z1.thr_theta.set(z1.thr_theta.get() * 0)
            z2.thr_theta.set(z2.thr_theta.get() * 0)

    def infer(self, x):
        """
        Processes (inference-only) an observation (sensory stimulus pattern) for a fixed
        stimulus window time T and produces an action spike trains.

        Args:
            x: observed pattern to input to this spiking neural model

        Returns:
            action spike train
        """
        self.clamp(x)
        
        inputs = jnp.array(self.advance_proc.pack_rows(self.T, t=lambda xx: xx, dt=self.dt))
        stateManager.state, outputs = self.advance_proc.scan(inputs) ## outputs -> (v1, s1, v2, s2)
        action_spikes = jnp.squeeze(outputs[3], axis=1)
        return action_spikes

    def process(self, x, modulator=0.):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T and produces an action spike trains. This routine carries out
        learning alongside inference at each simulated time-step.

        Note that this model assumes batch sizes of one (online learning).

        Args:
            x: observed pattern to input to this spiking neural model

            modulator: <unused>

        Returns:
            action spike train
        """
        V1, V2, W1, W2, z0, z1, z2 = self.circuit.get_components("V1", "V2", "W1", "W2", "z0", "z1", "z2")

        self.clamp(x)
        W2.modulator.set(modulator)
        inputs = jnp.array(self.adapt_proc.pack_rows(self.T, t=lambda xx: xx, dt=self.dt))
        stateManager.state, outputs = self.adapt_proc.scan(inputs)
        v1, s1, v2, s2 = outputs
        action_spikes = jnp.squeeze(s2, axis=1)
        return action_spikes

    def adapt(self, reward, act=None):
        """
        Triggers a step of modulated learning (i.e., MS-STDP/MS-STDP-ET). In other words, this 
        routine applies a traced STDP update via a driving reward signal.

        Args:
            reward: current modulatory signal (i.e., reward/dopamine value) to drive MS-STDP-ET update.

            act: action mask to apply to control layer

        """
        W1, W2, z0, z1, z2 = self.circuit.get_components("W1", "W2", "z0", "z1", "z2")
        
        W2.modulator.set(reward)
        if act is not None:
            W2.outmask.set(act)
        self.modulated_evolve_proc.run(t=(self.T + 1) * self.dt, dt=self.dt)
        #W2.eligibility.set(W2.eligibility.value * 0)

        if self.normalize:
            self.norm()
