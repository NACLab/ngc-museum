from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats

def _run_cell(dt, targ, mu, mask):
    return _run_gaussian_cell(dt, targ, mu, mask)

def _run_gaussian_cell(dt, targ, mu, mask):
    #N = jnp.sum(mask)
    dmu = (targ - mu) * 2. * mask # e (error unit)
    dtarg = -dmu # reverse of e
    L = -jnp.sum(jnp.square(dmu)) #* 0.5
    #return dmu * (1./N), dtarg * (1./N), L * (1./N)
    return dmu, dtarg, L

class MaskedErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell

    def __init__(self, name, n_units, use_avg_loss=True, batch_size=1,
                 shape=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.use_avg_loss = use_avg_loss

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0.) # loss compartment
        self.mu = Compartment(restVals) # mean/mean name. input wire
        self.dmu = Compartment(restVals) # derivative mean
        self.target = Compartment(restVals) # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    @staticmethod
    def _advance_state(dt, use_avg_loss, mu, dmu, target, dtarget, modulator, mask):
        # dmu, dtarget, L = _run_cell(dt, target * mask, mu * mask)
        # dmu = dmu * modulator * mask
        # dtarget = dtarget * modulator * mask
        # mask = mask * 0. + 1.

        ## compute Gaussian error cell output
        N = 1.
        if use_avg_loss:
            N = jnp.sum(mask)
        dmu, dtarget, L = _run_cell(dt, target, mu, mask)
        dmu = dmu * (1. / N)
        dtarget = dtarget * (1. / N)
        L = L * (1. / N)

        dmu = dmu * modulator
        dtarget = dtarget * modulator
        mask = mask * 0. + 1. ## "eat" the mask as it should only apply at time t
        return dmu, dtarget, L, mask

    @resolver(_advance_state)
    def advance_state(self, dmu, dtarget, L, mask):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)
        self.mask.set(mask)

    @staticmethod
    def _reset(batch_size, shape): #n_units
        _shape = (batch_size, shape[0])
        if len(shape) > 1:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        restVals = jnp.zeros(_shape)
        dmu = restVals
        dtarget = restVals
        target = restVals
        mu = restVals
        modulator = mu + 1.
        L = 0.
        mask = jnp.ones(_shape)
        return dmu, dtarget, target, mu, modulator, L, mask

    @resolver(_reset)
    def reset(self, dmu, dtarget, target, mu, modulator, L, mask):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(modulator)
        self.L.set(L)
        self.mask.set(mask)

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
