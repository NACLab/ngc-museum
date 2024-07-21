from ngclearn import Context, numpy as jnp
import math
from ngclearn.components import (LIFCell, PoissonCell, StaticSynapse,
                                 VarTrace, Monitor, EventSTDPSynapse, TraceSTDPSynapse)
from custom.ti_stdp_synapse import TI_STDP_Synapse
from ngclearn.operations import summation
from jax import jit, random
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import viz_block
from ngclearn.utils.model_utils import scanner
import ngclearn.utils.weight_distribution as dists
from ngclearn.utils.model_utils import normalize_matrix
from matplotlib import pyplot as plt
import ngclearn.utils as utils

def get_nodes(model):
    nodes = model.get_components("W1", "W1ie", "W1ei", "z0", "z1e", "z1i",
                                 "W2", "W2ie", "W2ei", "z2e", "z2i", "M")
    map = {} ## node-name look-up hash table
    for node in nodes:
        map[node.name] = node
    return nodes, map

def build_model(seed=1234, in_dim=1, is_patch_model=False, algo="tistdp"):
    window_length = 250 #300

    dt = 1
    X_size = int(jnp.sqrt(in_dim))
    # X_size = 28
    # in_dim = (X_size * X_size)
    if is_patch_model:
        hidden_size = 10 * 10
        out_size = 6 * 6
    else:
        hidden_size = 25 * 25
        out_size = 15 * 15

    exc = 22.5
    inh = 120. #60. #15. #10.
    tau_m_e = 100. # ms (excitatory membrane time constant)
    tau_m_i = 100. # ms (inhibitory membrane time constant)
    tau_theta = 1e5
    theta_plus = 0.05
    thr_jitter = 0.

    R1 = R2 = 1.
    if algo == "tistdp":
        R1 = 1.
        R2 = 6. #1.
    elif algo == "evstdp":
        R1 = 1.  # 6.
        R2 = 6.  # 12.
    elif algo == "trstdp":
        R1 = 1.  # 6.
        R2 = 6.  # 12.
        tau_theta = 1e4 #1e5
        theta_plus = 0.2 #0.05

    px = py = X_size
    hidx = hidy = int(jnp.sqrt(hidden_size))

    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 12)

    with Context("model") as model:
        M = Monitor("M", default_window_length=window_length)
        ## layer 0
        z0 = PoissonCell("z0", n_units=in_dim, max_freq=63.75, key=subkeys[0])
        ## layer 1
        z1e = LIFCell("z1e", n_units=hidden_size, tau_m=tau_m_e, resist_m=tau_m_e/dt,
                      refract_time=5., thr_jitter=thr_jitter, tau_theta=tau_theta,
                      theta_plus=theta_plus, one_spike=True, key=subkeys[2])
        z1i = LIFCell("z1i", n_units=hidden_size, tau_m=tau_m_i, resist_m=tau_m_i/dt,
                      refract_time=5., thr_jitter=thr_jitter, thr=-40., v_rest=-60.,
                      v_reset=-45., tau_theta=0.)
        W1ie = StaticSynapse("W1ie", shape=(hidden_size, hidden_size),
                             weight_init=dists.hollow(-inh))
        W1ei = StaticSynapse("W1ei", shape=(hidden_size, hidden_size),
                             weight_init=dists.eye(exc))
        ## layer 2
        z2e = LIFCell("z2e", n_units=out_size, tau_m=tau_m_e, resist_m=tau_m_e/dt,
                      refract_time=5., thr_jitter=thr_jitter, tau_theta=tau_theta,
                      theta_plus=theta_plus, one_spike=True, key=subkeys[4])
        z2i = LIFCell("z2i", n_units=out_size, tau_m=tau_m_i, resist_m=tau_m_i/dt,
                      refract_time=5., thr_jitter=thr_jitter, thr=-40., v_rest=-60.,
                      v_reset=-45., tau_theta=0.)
        W2ie = StaticSynapse("W2ie", shape=(out_size, out_size),
                             weight_init=dists.hollow(-inh))
        W2ei = StaticSynapse("W2ei", shape=(out_size, out_size),
                             weight_init=dists.eye(exc))

        tr0 = tr1 = tr2 = None
        if algo == "tistdp":
            print(" >> Equipping SNN with TI-STDP Adaptation")
            W1 = TI_STDP_Synapse("W1", alpha=0.0075 * 0.5, beta=1.25,
                                 pre_decay=0.75,
                                 shape=((X_size ** 2), hidden_size),
                                 weight_init=dists.uniform(amin=0.025,
                                                           amax=0.8),
                                 resist_scale=R1, key=subkeys[1])
            W2 = TI_STDP_Synapse("W2", alpha=0.025 * 2, beta=2,
                                 pre_decay=0.25 * 0.5,
                                 shape=(hidden_size, out_size),
                                 weight_init=dists.uniform(amin=0.025,
                                                           amax=0.8),
                                 resist_scale=R2, key=subkeys[6])
            # wire relevant plasticity statistics to synaptic cables W1 and W2
            W1.pre << z0.tols
            W1.post << z1e.tols
            W2.pre << z1e.tols
            W2.post << z2e.tols
        elif algo == "trstdp":
            print(" >> Equipping SNN with Trace/TR-STDP Adaptation")
            tau_tr = 20.  # 10.
            trace_delta = 0.
            x_tar1 = 0.3
            x_tar2 = 0.025
            Aplus = 1e-2  ## LTP learning rate (STDP); nu1
            Aminus = 1e-3  ## LTD learning rate (STDP); nu0
            mu = 1.  # 0.
            W1 = TraceSTDPSynapse("W1", shape=(in_dim, hidden_size), mu=mu,
                                  A_plus=Aplus, A_minus=Aminus, eta=1.,
                                  pretrace_target=x_tar1,
                                  weight_init=dists.uniform(amin=0.025, amax=0.8),
                                  resist_scale=R1, key=subkeys[1])
            W2 = TraceSTDPSynapse("W2", shape=(hidden_size, out_size), mu=mu,
                                  A_plus=Aplus, A_minus=Aminus, eta=1.,
                                  pretrace_target=x_tar2, resist_scale=R2,
                                  weight_init=dists.uniform(amin=0.025, amax=0.8),
                                  key=subkeys[3])
            ## traces
            tr0 = VarTrace("tr0", n_units=in_dim, tau_tr=tau_tr,
                           decay_type="exp",
                           a_delta=trace_delta)
            tr1 = VarTrace("tr1", n_units=hidden_size, tau_tr=tau_tr,
                           decay_type="exp",
                           a_delta=trace_delta)
            tr2 = VarTrace("tr2", n_units=out_size, tau_tr=tau_tr,
                           decay_type="exp",
                           a_delta=trace_delta)
            # wire relevant compartment statistics to synaptic cables W1 and W2
            W1.preTrace << tr0.trace
            W1.preSpike << z0.outputs
            W1.postTrace << tr1.trace
            W1.postSpike << z1e.s

            W2.preTrace << tr1.trace
            W2.preSpike << z1e.s
            W2.postTrace << tr2.trace
            W2.postSpike << z2e.s
        elif algo == "evstdp":
            print(" >> Equipping SNN with EV-STDP Adaptation")
            ## EV-STDP meta-parameters
            eta_w = 1.
            Aplus1 = 0.0055
            Aminus1 = 0.25 * 0.0055
            Aplus2 = 0.0055
            Aminus2 = 0.05 * 0.0055
            lmbda = 0.  # 0.01
            W1 = EventSTDPSynapse("W1", shape=(in_dim, hidden_size), eta=eta_w,
                                  A_plus=Aplus1, A_minus=Aminus1,
                                  lmbda=lmbda, w_bound=1., presyn_win_len=2.,
                                  weight_init=dists.uniform(0.025, 0.8),
                                  resist_scale=R1, key=subkeys[1])
            W2 = EventSTDPSynapse("W2", shape=(hidden_size, out_size),
                                  eta=eta_w,
                                  A_plus=Aplus2, A_minus=Aminus2,
                                  lmbda=lmbda, w_bound=1., presyn_win_len=2.,
                                  weight_init=dists.uniform(amin=0.025, amax=0.8),
                                  resist_scale=R2, key=subkeys[6])
            # wire relevant plasticity statistics to synaptic cables W1 and W2
            W1.pre_tols << z0.tols
            W1.postSpike << z1e.s
            W2.pre_tols << z1e.tols
            W2.postSpike << z2e.s
        else:
            print("ERROR: algorithm ", algo, " provided is not supported!")
            exit()

        ## layer 0 to layer 1
        W1.inputs << z0.outputs
        W1ie.inputs << z1i.s
        z1e.j << summation(W1.outputs, W1ie.outputs)
        W1ei.inputs << z1e.s
        z1i.j << W1ei.outputs
        ## layer 1 to layer 2
        W2.inputs << z1e.s_raw
        W2ie.inputs << z2i.s
        z2e.j << summation(W2.outputs, W2ie.outputs)
        W2ei.inputs << z2e.s
        z2i.j << W2ei.outputs

        ## wire statistics into global monitor
        ## layer 1 stats
        M << z1e.s
        M << z1e.j
        M << z1e.v
        ## layer 2 stats
        M << z2e.s
        M << z2e.j
        M << z2e.v

        if algo == "trstdp":
            advance, adv_args = model.compile_by_key(
                W1, W1ie, W1ei, W2, W2ie, W2ei,
                z0, z1e, z1i, z2e, z2i,
                tr0, tr1, tr2, M,
                compile_key="advance_state")
            evolve, evolve_args = model.compile_by_key(W1, W2,
                                                       compile_key="evolve")
            reset, reset_args = model.compile_by_key(
                z0, z1e, z1i, z2e, z2i, W1, W2, W1ie, W1ei, W2ie, W2ei, tr0,
                tr1, tr2,
                compile_key="reset")
        else:
            advance, adv_args = model.compile_by_key(
                W1, W1ie, W1ei, W2, W2ie, W2ei,
                z0, z1e, z1i, z2e, z2i, M,
                compile_key="advance_state")
            evolve, evolve_args = model.compile_by_key(W1, W2, compile_key="evolve")
            reset, reset_args = model.compile_by_key(
                z0, z1e, z1i, z2e, z2i, W1, W2, W1ie, W1ei, W2ie, W2ei,
                compile_key="reset")
            model.wrap_and_add_command(jit(model.reset), name="reset")

        @model.dynamicCommand
        def clamp(x):
            z0.inputs.set(x)

        @model.dynamicCommand
        def viz(name, low_rez=True, raster_name="exp/raster_plot"):
            viz_block([W1.weights.value, W2.weights.value],
                      [(px, py), (hidx, hidy)], name + "_block", padding=2,
                      low_rez=low_rez)

            fig, ax = plt.subplots(3, 2, sharex=True, figsize=(15, 8))

            for i in range(out_size):
                ax[1][0].plot([i for i in range(window_length)],
                              M.view(z1e.v)[:, :, i])
                ax[0][0].plot([i for i in range(window_length)],
                              M.view(z1e.j)[:, :, i])

                ax[1][1].plot([i for i in range(window_length)],
                              M.view(z2e.v)[:, :, i])
                ax[0][1].plot([i for i in range(window_length)],
                              M.view(z2e.j)[:, :, i])
            utils.viz.raster.create_raster_plot(M.view(z1e.s), ax=ax[2][0])
            utils.viz.raster.create_raster_plot(M.view(z2e.s), ax=ax[2][1])
            # plt.show()
            plt.savefig(raster_name)
            plt.close()

        @scanner
        def observe(current_state, args):
            _t, _dt = args
            current_state = model.advance_state(current_state, t=_t, dt=_dt)
            current_state = model.evolve(current_state, t=_t, dt=_dt)
            return current_state, (current_state[z1e.s_raw.path],
                                   current_state[z2e.s_raw.path])

        @model.dynamicCommand
        def showStats(i):
            print(f"\n~~~~~Iteration {str(i)}~~~~~~")
            print("W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(W1.weights.value), jnp.amax(W1.weights.value), jnp.mean(W1.weights.value),
                                                                        jnp.linalg.norm(W1.weights.value)))
            print("W2:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(jnp.amin(W2.weights.value), jnp.amax(W2.weights.value), jnp.mean(W2.weights.value),
                                                                        jnp.linalg.norm(W2.weights.value)))
    #M.halt_all()
    return model

def load_from_disk(model_directory, param_dir="/custom", disable_adaptation=True):
    with Context("model") as model:
        model.load_from_dir(model_directory, custom_folder=param_dir)
        nodes = model.get_components("W1", "W1ie", "W1ei", "W2", "W2ie", "W2ei",
                                     "z0", "z1e", "z1i", "z2e", "z2i")
        (W1, W1ie, W1ei, W2, W2ie, W2ei,z0, z1e, z1i, z2e, z2i) = nodes
        if disable_adaptation:
            z1e.tau_theta = 0. ## disable homeostatic adaptation
            z2e.tau_theta = 0. ## disable homeostatic adaptation

        advance, adv_args = model.compile_by_key(
            W1, W1ie, W1ei, W2, W2ie, W2ei,
            z0, z1e, z1i, z2e, z2i,
            compile_key="advance_state")
        evolve, evolve_args = model.compile_by_key(W1, W2, compile_key="evolve")
        reset, reset_args = model.compile_by_key(
            z0, z1e, z1i, z2e, z2i, W1, W2, W1ie, W1ei, W2ie, W2ei,
            compile_key="reset")
        model.wrap_and_add_command(jit(model.reset), name="reset")

        @model.dynamicCommand
        def clamp(x):
            z0.inputs.set(x)

        @model.dynamicCommand
        def viz(name, low_rez=True):
            viz_block([W1.weights.value, W2.weights.value],
                      [(28, 28), (10, 10)], name + "_block", padding=2,
                      low_rez=low_rez)

        @scanner
        def infer(current_state, args):
            _t, _dt = args
            current_state = model.advance_state(current_state, t=_t, dt=_dt)
            return current_state, (current_state[z1e.s_raw.path],
                                   current_state[z2e.s_raw.path])

        @model.dynamicCommand
        def showStats(i):
            print(f"\n~~~~~Iteration {str(i)}~~~~~~")
            print("W1:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
                jnp.amin(W1.weights.value), jnp.amax(W1.weights.value),
                jnp.mean(W1.weights.value),
                jnp.linalg.norm(W1.weights.value)))
            print("W2:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
                jnp.amin(W2.weights.value), jnp.amax(W2.weights.value),
                jnp.mean(W2.weights.value),
                jnp.linalg.norm(W2.weights.value)))
    return model

