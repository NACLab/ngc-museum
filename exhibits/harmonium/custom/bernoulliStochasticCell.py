# %%

from jax import numpy as jnp, random, jit

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.model_utils import create_function, sigmoid
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2, step_rk4
from ngcsimlib.logger import info



class BernoulliStochasticCell(JaxComponent): ## Bernoulli stochastic cell

    def __init__(
            self, name, n_units, is_stoch=True, batch_size=1, resist_scale=1., shape=None, **kwargs
    ):
        super().__init__(name, **kwargs)

        #self.act_fx = "sigmoid"

        ## Layer size setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.is_stoch = is_stoch
        self.batch_size = batch_size

        restVals = jnp.zeros(_shape)
        self.inputs = Compartment(restVals, display_name="Input Stimulus")  # injected input
        self.p = Compartment(restVals, display_name="State Probability")
        self.s = Compartment(restVals, display_name="State Sample")

    @compilable
    def advance_state(self, dt):
        ## Get the compartment values
        inputs = self.inputs.get()
        
        key, skey = random.split(self.key.get(), 2)
        states = prob = inputs
        if self.is_stoch:
            prob = sigmoid(inputs) ## compute p(s=1 | inputs)
            ## s ~ p(s=1 | inputs)
            states = random.bernoulli(skey, p=prob, shape=prob.shape) * 1. ## sample Bernoulli neurons

        ## advance state of compartments
        self.key.set(key)
        self.p.set(prob)
        self.s.set(states)

    @compilable
    def reset(self):  # , batch_size, shape): #n_units
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape)
        self.p.set(restVals)
        self.s.set(restVals)
