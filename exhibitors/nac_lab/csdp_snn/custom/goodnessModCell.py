from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit, nn
from jax import grad, value_and_grad #, jacfwd
from functools import partial
from ngclearn.utils import tensorstats

@partial(jit, static_argnums=[2])
def calc_goodness(z, thr, maximize=True):
    z_sqr = jnp.square(z) 
    delta = jnp.sum(z_sqr, axis=1, keepdims=True)
    if maximize:
        ## maximize for positive samps, minimize for negative samps
        delta = delta - thr
    else:
        ## minimize for positive samps, maximize for negative samps
        delta = -delta + thr
    scale = 1. 
    delta = delta * scale 
    # gets the probability P(pos)
    p = nn.sigmoid(delta)
    eps = 1e-5 #1e-6
    p = jnp.clip(p, eps, 1.0 - eps)
    return p, delta

@partial(jit, static_argnums=[3])
def calc_loss(z, lab, thr, keep_batch=False):
    _lab = (lab > 0.).astype(jnp.float32)
    p, logit = calc_goodness(z, thr)
    CE = jnp.maximum(logit, 0) - logit * _lab + jnp.log(1. + jnp.exp(-jnp.abs(logit)))
    L = jnp.sum(CE, axis=1, keepdims=True)
    if keep_batch == False:
        L = jnp.mean(L) 
    return L

@partial(jit, static_argnums=[3])
def calc_mod_signal(z, lab, thr, keep_batch):
    L, d_z = value_and_grad(calc_loss, argnums=0)(z, lab, thr, keep_batch)
    return L, d_z

class GoodnessModCell(JaxComponent):
    # The proposed contrastive / goodness modulator; this cell produces a
    # signal based on a constrastive threshold-based functonal, i.e., "goodness",
    # which produces a modulatory value equal to the first derivative of the
    # contrastive functional.

    def __init__(self, name, n_units, threshold=7., use_dyn_threshold=False,
                 batch_size=1, shape=None,
                 **kwargs):
        super().__init__(name, **kwargs)

        self.goodness_threshold = threshold
        self.use_dyn_threshold = use_dyn_threshold ## this overrides threshold if True

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size

        ## (default) Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.contrastLabels = Compartment(jnp.zeros((self.batch_size, 1)))
        self.inputs = Compartment(restVals)
        self.loss = Compartment(0.) # loss compartment
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed

    @staticmethod
    def _advance_state(use_dyn_threshold, goodness_threshold, inputs, contrastLabels):
        thr_theta = goodness_threshold
        if use_dyn_threshold:
            thr_theta = jnp.mean(inputs, axis=1, keepdims=True)
        keep_batch = False
        loss, modulator = calc_mod_signal(
            inputs, contrastLabels, thr_theta, keep_batch
        )
        return modulator, loss

    @resolver(_advance_state)
    def advance_state(self, modulator, loss):
        self.modulator.set(modulator)
        self.loss.set(loss)

    @staticmethod
    def _reset(batch_size, shape): #n_units
        _shape = (batch_size, shape[0])
        if len(shape) > 1:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        restVals = jnp.zeros(_shape)
        inputs = restVals
        contrastLabels = restVals
        modulator = restVals
        loss = 0.
        return inputs, contrastLabels, modulator, loss

    @resolver(_reset)
    def reset(self, inputs, contrastLabels, modulator, loss):
        self.inputs.set(inputs)
        self.contrastLabels.set(contrastLabels)
        self.modulator.set(modulator)
        self.loss.set(loss)

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

