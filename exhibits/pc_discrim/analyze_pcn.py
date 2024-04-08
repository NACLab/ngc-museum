from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse
## bring in model from museum
from ngcsimlib.controller import Controller
## bring in ngc-learn analysis tools
from ngclearn.utils.model_utils import measure_ACC, measure_CatNLL

from train_pcn import eval_model

dataX = "../data/mnist/testX.npy"
dataY = "../data/mnist/testY.npy"
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
print("=> Data X: {} | Y: {}".format(dataX, dataY))

_X = jnp.load(dataX)
_Y = jnp.load(dataY)

model = Controller()
model.load_from_dir(model_directory="exp/pcn")

nll, acc = eval_model(model, _X, _Y, mb_size=1000)

print("=> NLL = {}  Acc = {}".format(nll, acc))
