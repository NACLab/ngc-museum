#from ngcsimlib.controller import Controller
from jax import numpy as jnp, random, nn, jit
import sys
from pc_model import PCN


dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

n_iter = 100

in_dim = 3
Xb = random.uniform(subkeys[0], (1, in_dim), dtype=jnp.float32)
out_dim = 2
Yb = random.uniform(subkeys[0], (1, out_dim), minval=0., maxval=0.6, dtype=jnp.float32)
Yb = Yb * jnp.asarray([[-1, 1.]])

model = PCN(subkeys[1], in_dim, out_dim, hid1_dim=16, hid2_dim=16, T=20,
            dt=1., tau_m=20., act_fx="lrelu", exp_dir="exp", model_name="pcn")

for i in range(n_iter):
    yMu_0, yMu = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
    print("---")
    print("mu: ",yMu_0)
    print(" y: ",Yb)
