from ngclearn import Component, Compartment, resolver
from .LCNSynapse import LCNSynapse
from ngclearn.utils.model_utils import normalize_matrix

from jax import numpy as jnp, random
import os.path


class TI_STDP_LCNSynapse(LCNSynapse):
    def __init__(self, name, shape, model_shape=(1, 1), alpha=0.0075,
                 beta=0.5, pre_decay=0.5, resist_scale=1., p_conn=1.,
                 weight_init=None, Aplus=1, Aminus=1, **kwargs):
        super().__init__(name, shape=shape, model_shape=model_shape,
                         weight_init=weight_init, bias_init=None,
                         resist_scale=resist_scale, p_conn=p_conn, **kwargs)

        # Params
        self.batch_size = 1
        self.alpha = alpha
        self.beta = beta
        self.pre_decay = pre_decay
        self.Aplus = Aplus
        self.Aminus = Aminus

        # Compartments
        self.pre = Compartment(None)
        self.post = Compartment(None)

        self.reset()

    @staticmethod
    def _norm(weights, norm_scale):
        return normalize_matrix(weights, wnorm=100, scale=norm_scale)

    @resolver(_norm)
    def norm(self, weights):
        self.weights.set(weights)

    @staticmethod
    def _reset(batch_size, shape):
        return jnp.zeros((batch_size, shape[0])), \
            jnp.zeros((batch_size, shape[1])), \
            jnp.zeros((batch_size, shape[0])), \
            jnp.zeros((batch_size, shape[1]))

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)

    @staticmethod
    def _evolve(pre, post, weights,
                shape, alpha, beta, pre_decay, Aplus, Aminus,
                t, dt):
        mask = jnp.where(0 != jnp.abs(weights), 1., 0.)

        pre_size = shape[0]
        post_size = shape[1]

        pre_synaptic_binary_mask = jnp.where(pre > 0, 1., 0.)
        post_synaptic_binary_mask = jnp.where(post > 0, 1., 0.)

        broadcast_pre_synaptic_binary_mask = jnp.zeros(shape) + jnp.reshape(
            pre_synaptic_binary_mask, (pre_size, 1))
        broadcast_post_synaptic_binary_mask = jnp.zeros(shape) + jnp.reshape(
            post_synaptic_binary_mask, (1, post_size))

        broadcast_pre_synaptic_time = jnp.zeros(shape) + jnp.reshape(pre, (
        pre_size, 1))
        broadcast_post_synaptic_time = jnp.zeros(shape) + jnp.reshape(post, (
        1, post_size))

        no_pre_synaptic_spike_mask = broadcast_post_synaptic_binary_mask * (
                1. - broadcast_pre_synaptic_binary_mask)
        no_pre_synaptic_weight_update = (-alpha) * (pre_decay / jnp.exp(
            (1 / dt) * (t - broadcast_post_synaptic_time)))
        # no_pre_synaptic_weight_update = no_pre_synaptic_weight_update

        # Both have spiked
        both_spike_mask = (broadcast_post_synaptic_binary_mask *
                           broadcast_pre_synaptic_binary_mask)
        both_spike_update = (-alpha / (
                broadcast_pre_synaptic_time - broadcast_post_synaptic_time - (
                    0.5 * dt))) * \
                            (beta / jnp.exp(
                                (1 / dt) * (t - broadcast_post_synaptic_time)))

        masked_no_pre_synaptic_weight_update = (no_pre_synaptic_spike_mask *
                                                no_pre_synaptic_weight_update)
        masked_both_spike_update = both_spike_mask * both_spike_update

        plasticity = jnp.where(masked_both_spike_update > 0,
                               Aplus * masked_both_spike_update,
                               Aminus * masked_both_spike_update)

        decay = masked_no_pre_synaptic_weight_update

        plasticity = plasticity * (1 - weights)

        decay = decay * weights

        update = plasticity + decay
        _W = weights + update
        ## return masked synaptic weight matrix (enforced structure)
        return jnp.clip(_W, 0., 1.) * mask

    @resolver(_evolve)
    def evolve(self, weights):
        self.weights.set(weights)
