from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.operations import summation
from jax import numpy as jnp, random, jit
from ngclearn.components import StaticSynapse, MSTDPETSynapse, VarTrace, PoissonCell
from custom import NoisyLIFCell as LIFCell
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import normalize_matrix, softmax, one_hot

def init_sparse_syn(dkey, shape, n_minus, n_plus, is_unique=False): ## synapse initialization scheme
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
        tau_m_e = 100.  #100.5
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
                "W1", shape=(in_dim, n_hid), weight_init=dist.uniform(-1., 1.), resist_scale=4.,
                key=subkeys[0]
            )
            W1.weights.set(
                init_sparse_syn(subkeys[0], W1.weights.value.shape, n_minus, n_plus))  # jnp.sign(W1.weights.value))
            V1 = StaticSynapse( ## lateral inhibitory synapses for feature layer
                "V1", shape=(n_hid, n_hid), weight_init=dist.hollow(scale=1.), resist_scale=-inhR, key=subkeys[1]
            )
            #V1.weights.set(V1.weights.value * random.uniform(subkeys[1], shape=(n_hid, n_hid), minval=0.4, maxval=1.))
            z1 = LIFCell( ## projection/hidden layer
                "z1", n_units=n_hid, tau_m=tau_m_e, resist_m=1., v_decay=1., thr=-52., v_rest=-65., v_reset=-65.,
                eps_scale=lif_noise_eps, tau_theta=tau_theta, theta_plus=theta_plus, refract_time=0., enforce_wta=False,
                v_min=None, key=subkeys[2]
            )
            ## hidden-to-place synapses
            V2 = StaticSynapse( ## lateral inhibitory synapses for control layer
                "V2", shape=(out_dim, out_dim), weight_init=dist.hollow(scale=1.), resist_scale=-inhR2, key=subkeys[3]
            )
            #V2.weights.set(V2.weights.value * random.uniform(subkeys[3], shape=(out_dim, out_dim), minval=0.4, maxval=1.))
            W2 = MSTDPETSynapse(
                "W2", shape=(n_hid, out_dim), A_plus=Aplus2, A_minus=Aminus2, tau_elg=tau_e2, eta=nu2, tau_w=tau_w2,
                pretrace_target=x_tar2, weight_init=dist.uniform(0.01, w_max_init), resist_scale=1., w_bound=1., key=subkeys[4]
            )
            z2 = LIFCell(
                "z2", n_units=out_dim, tau_m=tau_m_e, resist_m=1., v_decay=1., thr=-52., v_rest=-65., v_reset=-65.,
                eps_scale=lif_noise_eps, tau_theta=tau_theta, theta_plus=theta_plus, refract_time=0., enforce_wta=False,
                v_min=None, key=subkeys[5]
            )
            ## set up traces
            W2_p1_tr = VarTrace(
                "W2_p1_tr", n_units=n_hid, tau_tr=tau_p, decay_type="exp", P_scale=p_minus, a_delta=0., key=subkeys[0]
            )
            W2_p2_tr = VarTrace(
                "W2_p2_tr", n_units=out_dim, tau_tr=tau_p, decay_type="exp", P_scale=p_minus, a_delta=0., key=subkeys[0]
            )

            ## wire z0 to z1
            W1.inputs << z0.outputs
            V1.inputs << z1.s
            z1.j << summation(W1.outputs, V1.outputs)
            ## wire z1 to z2
            W2.inputs << z1.s
            V2.inputs << z2.s
            z2.j << summation(W2.outputs, V2.outputs)
            # wire cells to their respective traces
            W2_p1_tr.inputs << z1.s ## for W2's plasticity
            W2_p2_tr.inputs << z2.s ## for W2's plasticity

            # wire relevant compartment statistics to synaptic cables W1 and W2 (to drive learning rules)
            W2.preTrace << W2_p1_tr.trace
            W2.preSpike << z1.s
            W2.postTrace << W2_p2_tr.trace
            W2.postSpike << z2.s

            reset_cmd, reset_args = self.circuit.compile_by_key(
                V1, V2, W1, W2, z0, z1, z2, W2_p1_tr, W2_p2_tr,
                compile_key="reset"
            )

            advance_cmd, advance_args = self.circuit.compile_by_key(
                V1, V2, W1, W2, z0, z1, z2, W2_p1_tr, W2_p2_tr,
                compile_key="advance_state"
            )

            _evolve_cmd, _evolve_args = self.circuit.compile_by_key(W2, compile_key="evolve")
            evolve_cmd, evolve_args = self.circuit.compile_by_key(W2, compile_key="evolve", name="modulated_evolve")
            self._dynamic()


    def _dynamic(self):
        W1, W2, z0, z1, z2 = self.circuit.get_components(
            "W1", "W2", "z0", "z1", "z2"
        )

        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        #self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")
        self.circuit.add_command(wrap_command(jit(self.circuit.modulated_evolve)), name="modulated_evolve")
        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")

        @Context.dynamicCommand
        def norm():
            W2.weights.set(normalize_matrix(W2.weights.value, self.wnorm2, order=1, axis=0))

        @Context.dynamicCommand
        def clamp(x):
            z0.inputs.set(x)
            #W1.inputs.set(x)

        @scanner
        def infer(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
            return (compartment_values,
                    (compartment_values[z2.v.path], compartment_values[z2.s.path]))

        @scanner
        def process(compartment_values, args):
            _t, _dt = args
            compartment_values = self.circuit.advance_state(compartment_values, t=_t, dt=_dt)
            compartment_values = self.circuit.evolve(compartment_values, t=_t, dt=_dt)
            return (compartment_values,
                    (compartment_values[z1.v.path], compartment_values[z1.s.path],
                     compartment_values[z2.v.path], compartment_values[z2.s.path]))

        if self.normalize:
            self.circuit.norm()

    def set_to_resting_state(self):
        """
        Sets all internal states of this model to their resting values.
        """
        self.circuit.reset()
        z1, z2 = self.circuit.get_components("z1", "z2")
        if self.reset_volt_thresholds:
            z1.thr_theta.set(z1.thr_theta.value * 0)
            z2.thr_theta.set(z2.thr_theta.value * 0)

    def infer(self, x):
        """
        Processes (inference-only) an observation (sensory stimulus pattern) for a fixed
        stimulus window time T and produces an action spike trains.

        Args:
            x: observed pattern to input to this spiking neural model

        Returns:
            action spike train
        """
        self.circuit.clamp(x)
        v, s = self.circuit.infer(
            jnp.array([[self.dt * i, self.dt] for i in range(self.T)])
        )
        action_spikes = jnp.squeeze(s, axis=1)
        return action_spikes

    def process(self, x, modulator=0.):
        """
        Processes an observation (sensory stimulus pattern) for a fixed
        stimulus window time T and produces an action spike trains. This routine carries out
        learning as well as inference.

        Note that this model assumes batch sizes of one (online learning).

        Args:
            x: observed pattern to input to this spiking neural model

            modulator: <unused>

        Returns:
            action spike train
        """
        V1, V2, W1, W2, z0, z1, z2 = self.circuit.get_components("V1", "V2", "W1", "W2", "z0", "z1", "z2")

        self.circuit.clamp(x)
        W2.modulator.set(modulator)
        v1, s1, v2, s2 = self.circuit.process(
            jnp.array([[self.dt * i, self.dt] for i in range(self.T)])
        )
        action_spikes = jnp.squeeze(s2, axis=1)
        #act_prob = softmax(jnp.sum(action_spikes, axis=0, keepdims=True) * alpha_T)
        return action_spikes

    def adapt(self, reward, act=None):
        """
        Triggers a step of modulated learning (i.e., MS-STDP/MS-STDP-ET).

        Args:
            reward: current modulatory signal (i.e., reward/dopamine value) to drive MS-STDP-ET update.

            act: action mask to apply to control layer

        Returns:
            action spike train
        """
        W1, W2, z0, z1, z2 = self.circuit.get_components("W1", "W2", "z0", "z1", "z2")
        W2.modulator.set(reward)
        if act is not None:
            W2.outmask.set(act)
            #W2.outmask.set(z2.s.value)
        self.circuit.modulated_evolve(t=(self.T + 1) * self.dt, dt=self.dt)

        #W2.eligibility.set(W2.eligibility.value * 0)

        if self.normalize:
            self.circuit.norm()
