import sys, getopt as gopt, optparse, time
import matplotlib.pyplot as plt

from jax import numpy as jnp, random
from ngclearn.utils.metric_utils import measure_MAE, measure_RMSE
from ei_rnn import EI_RNN

"""
################################################################################
Continuous-time Excitatory-Inhibitory Recurrent Network (EI-RNN) Exhibit File:

Trains/fits a continuous-time excitatory-inhibitory recurrent neural network 
(EI-RNN) to a toy stream of data points (sampled from a noisy sinusoidal wave). 

Usage:
$ python sim_ei_rnn.py --n_hid=8 --n_iter=40 --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '', ["n_hid=", "n_iter=", "verbosity="]
)

verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints synaptic stats on I/O)
n_iter = 40 ## number training iterations / epochs
n_hid = 10 ## hidden population size (includes both excitatory and inhibitory neurons)
for opt, arg in options:
    if opt in ("--n_hid"):
        n_hid = int(arg.strip())
    elif opt in ("--n_iter"):
        n_iter = int(arg.strip())
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())

################################################################################
## set up synthetic time-series forecasting problem
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 3)

# predict sine wave
x = jnp.expand_dims(jnp.linspace(0, 10, 1000), axis=1) ## time-span
y = jnp.sin(x) + random.normal(subkeys[0], shape=x.shape) * 0.025 ## create noisy sine wave
t = y[1:] ## Goal: predict target values (t+1)
y = y[:-1] ## Input: input values (t)

################################################################################
## set up EI-RNN model
model = EI_RNN(subkeys[1], obs_dim=1, hid_dim=n_hid) ## Demo model: 8 exc, 2 inh neurons
model.save_to_disk()

################################################################################
## fit EI-RNN to temporal stream
if verbosity > 0:
    print("---- Initial Synapse Stats -----")
    print(model.get_synapse_stats())

z, yHat, L = model.process(y, t, adapt_synapses=False) ## run model w learning "disabled"
mae = measure_MAE(yHat, t)
rmse = measure_RMSE(yHat, t)
print(f"Initial.Test: Loss = {L:.4f}  MAE = {mae:.4f}  RMSE = {rmse:.4f}")

## run main training loop
for i in range(n_iter):
    z, yHat, L = model.process(y, t, adapt_synapses=True) ## process & adapt to stream
    print(f"{i}| Train: Loss = {L:.4f}")

if verbosity > 0:
    print("---- Final Synapse Stats -----")
    print(model.get_synapse_stats())
model.save_to_disk(params_only=True) ## save final model to disk

## measure loss with synaptic adaptation turned off
z, yHat, L = model.process(y, t, adapt_synapses=False) ## run model w learning "disabled"
mae = measure_MAE(yHat, t)
rmse = measure_RMSE(yHat, t)
print(f"Final.Test: Loss = {L:.4f}  MAE = {mae:.4f}  RMSE = {rmse:.4f}")

plt.scatter(x[0:x.shape[0]-1,0], t[:,0], color="red")
plt.scatter(x[0:x.shape[0]-1,0], yHat[:,0], color="blue")

## plot EI-RNN predictions against time series data values
plt.xlabel("X (Inputs)")
plt.ylabel("f(X) (Outputs)")
plt.title("EI-RNN Plot")
plt.savefig(f"{model.exp_dir}/analysis/eirnn_series.jpg")
