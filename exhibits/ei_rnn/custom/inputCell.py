from jax import numpy as jnp, random, jit

from ngclearn import compilable 
from ngclearn import Compartment 
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.logger import info


class InputCell(JaxComponent): ## Simple input / pass-through cell

    def __init__(
            self, name, n_units, scale_factor=1., batch_size=1, shape=None, **kwargs
    ):
        super().__init__(name, **kwargs)

        ## Layer size setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.scale_factor = scale_factor

        restVals = jnp.zeros(_shape)
        self.inputs = Compartment(restVals, display_name="Input Stimulus") ## injected input
        self.outputs = Compartment(restVals, display_name="Output Response")
        self.deriv = Compartment(restVals + 1.)

    @compilable
    def advance_state(self, dt):
        ## Get the compartment values
        inputs = self.inputs.get() * self.scale_factor
        self.outputs.set(inputs) ## input goes directly into output

    @compilable
    def reset(self): 
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)
        if not self.inputs.targeted:
            self.inputs.set(restVals)
        self.outputs.set(restVals)
        self.deriv.set(restVals + 1.)

