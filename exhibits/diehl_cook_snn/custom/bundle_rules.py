"""
A faster compartment add bundle rule is written here (otherwise, the 
default ngcsimlib non-jit-i-fied add gets used
"""
from jax import numpy as jnp, random, jit
#from functools import partial
import time, sys

@jit
def add(x, y): # jit-i-fied addition
    return x + y

def fast_add(component, value, destination_compartment): ## jit-i-fied bundle rule
    curr_val = component.compartments[destination_compartment]
    new_val = add(curr_val, value)
    component.compartments[destination_compartment] = new_val
