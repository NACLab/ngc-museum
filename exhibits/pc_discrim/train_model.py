#from ngcsimlib.controller import Controller
from jax import numpy as jnp, random, nn, jit
import sys
from pc_model import PCN
from ngclearn.utils.model_utils import calc_acc, measure_CatNLL

_X = jnp.load("data/baby_mnist/babyX.npy")
_Y = jnp.load("data/baby_mnist/babyY.npy")
# _X = jnp.load("/home/ago/Research/spiking-pff-learning/data/mnist/trainX.npy")
# _Y = jnp.load("/home/ago/Research/spiking-pff-learning/data/mnist/trainY.npy")
x_dim = _X.shape[1]
y_dim = _Y.shape[1]

n_iter = 100
mb_size = 10 #200 # 256
# std of init - 0.025
n_batches = int(_X.shape[0]/mb_size)

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build model
model = PCN(subkeys[1], x_dim, y_dim, hid1_dim=128, hid2_dim=128, T=10, # T=20 #hid=500
            dt=1., tau_m=10., act_fx="sigmoid", exp_dir="exp", model_name="pcn")

for i in range(n_iter):
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]
    Y = _Y[ptrs,:]

    ## begin epoch
    Y_mu = []
    n_samp_seen = 0
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)

        idx = j * mb_size #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]
        Yb = Y[idx: idx + mb_size,:]

        yMu_0, yMu = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        Y_mu.append(yMu_0)
        n_samp_seen += Yb.shape[0]
        print("\r {} processed ".format(n_samp_seen))
    print()
    Y_mu = jnp.concatenate(Y_mu, axis=0)
    nll = measure_CatNLL(Y_mu, Y)
    acc = calc_acc(Y_mu, Y)
    print("{}: Acc = {}".format(i, acc, nll))
