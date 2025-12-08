# %%

import jax
import pickle
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn

from ngclearn import compilable  # from ngcsimlib.parser import compilable
from ngclearn import Compartment  # from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses.hebbian import HebbianSynapse
from ngclearn.utils import tensorstats
from ngcsimlib import deprecate_args
from ngclearn.utils.io_utils import save_pkl, load_pkl

@partial(jit, static_argnums=[1,2])
def _enforce_constraints(W, w_bound, is_nonnegative=True):
    """
    Enforces constraints that the (synaptic) efficacies/values within matrix
    `W` must adhere to.

    Args:
        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W
    if w_bound > 0.:
        if is_nonnegative:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

class TracedHebbianSynapse(HebbianSynapse):

    def __init__(
            self, name, shape, eta=0., weight_init=None, bias_init=None, w_bound=1., is_nonnegative=False,
            prior=("constant", 0.), w_decay=0., sign_value=1., optim_type="sgd", pre_wght=1., post_wght=1.,
            p_conn=1., resist_scale=1., tau_elg=0., mask=None, batch_size=1, **kwargs
    ):
        super().__init__(
            name, shape=shape, eta=eta, weight_init=weight_init, bias_init=bias_init, w_bound=w_bound,
            is_nonnegative=is_nonnegative, prior=prior, w_decay=w_decay, sign_value=sign_value, optim_type=optim_type,
            pre_wght=pre_wght, post_wght=post_wght, p_conn=p_conn, resist_scale=resist_scale, batch_size=batch_size,
            **kwargs
        )
        self.mask = 1.
        if mask is not None:
            self.mask = mask

        self.tau_elg = tau_elg
        self.elgWeights = Compartment(jnp.zeros(shape))
        self.elgBiases = Compartment(jnp.zeros(shape[1]))

    @compilable
    def advance_state(self):
        weights = self.weights.get()
        weights = weights * self.mask
        self.outputs.set((jnp.matmul(self.inputs.get(), weights) * self.resist_scale) + self.biases.get())

    @compilable
    def evolve(self, dt):
        # Get the variables
        pre = self.pre.get()
        post = self.post.get()
        weights = self.weights.get()
        biases = self.biases.get()
        opt_params = self.opt_params.get()

        ## calculate synaptic update values
        dWeights, dBiases = HebbianSynapse._compute_update(
            self.w_bound, self.is_nonnegative, self.sign_value, self.prior_type, self.prior_lmbda, self.pre_wght,
            self.post_wght,
            pre, post, weights
        )
        if self.tau_elg > 0.:
            self.dWeights.set(dWeights)
            elgWeights = self.elgWeights.get()
            elgWeights = elgWeights + (-elgWeights + dWeights) * dt / self.tau_elg
            self.elgWeights.set(elgWeights)
            dWeights = elgWeights

            if self.bias_init != None:
                self.dBiases.set(dBiases)
                elgBiases = self.elgBiases.get()
                elgBiases = elgBiases + (-elgBiases + dBiases) * dt / self.tau_elg
                self.elgBiases.set(elgBiases)
                dBiases = elgBiases

        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if self.bias_init != None:
            opt_params, [weights, biases] = self.opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = self.opt(opt_params, [weights], [dWeights])
        ## ensure synaptic efficacies adhere to constraints
        weights = _enforce_constraints(weights, self.w_bound, is_nonnegative=self.is_nonnegative)

        # Update compartments
        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        if self.tau_elg <= 0.:
            self.dWeights.set(dWeights)
            self.dBiases.set(dBiases)

    @compilable
    def reset(self):  # , batch_size, shape):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)  # outputs
        self.pre.set(preVals)  # pre
        self.post.set(postVals)  # post
        self.dWeights.set(jnp.zeros(self.shape))  # dW
        self.dBiases.set(jnp.zeros(self.shape[1]))  # db
        if self.tau_elg > 0.:
            self.elgWeights.set(jnp.zeros(self.shape))
            self.elgBiases.set(jnp.zeros(self.shape[1]))
