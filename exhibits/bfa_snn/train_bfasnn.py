from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time
from bfasnn_model import BFA_SNN as Model ## bring in model from museum
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

"""
################################################################################
BFA-Trained Spiking Neural Network (BFA-SNN) Exhibit File:

Fits a BFA-SNN (an SNN trained with broadcast feedback alignment) classifier
to the MNIST database.

Usage:
$ python train_bfasnn.py --dataX="/path/to/train_patterns.npy" \
                         --dataY="/path/to/train_labels.npy" \
                         --devX="/path/to/dev_patterns.npy" \
                         --devY="/path/to/dev_labels.npy" \
                         --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=", "verbosity="]
                                 )
# external dataset arguments
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
devX = "../../data/mnist/validX.npy"
devY = "../../data/mnist/validY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

lab_estimator = "current" # "voltage" # "spike"
n_iter = 30 ## number discrete time steps to simulate
mb_size = 250
n_batches = int(_X.shape[0]/mb_size)
save_point = 5 ## save model params every epoch/iteration modulo "save_point"

## set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build/configure BFA-SNN model
hid_dim = 1000
T = 100 ## number discrete time steps to simulate
dt = 0.25 ## integration time constant (set in accordance w/ Samadi et al., 2017)
tau_mem = 20. ## membrane potential time constant (set as in Samadi et al., 2017)
## Note: another way to calc "T" is to list out a time-span (in ms) and divide by dt
print("--- Building Model ---")
model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, T=T, dt=dt, tau_m=tau_mem)
model.save_to_disk() # save initial state of synapses to disk
print("--- Starting Simulation ---")

def eval_model(model, Xdev, Ydev, mb_size, verbosity=1): ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/mb_size)

    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size,:]
        Yb = Ydev[idx: idx + mb_size,:]
        ## run model inference
        _S, yMu, yCnt = model.process(obs=Xb, lab=Yb, adapt_synapses=False,label_dist_estimator=lab_estimator)
        ## record metric measurements
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0] ## un-normalize score
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0] ## un-normalize score
        nll += _nll
        acc += _acc

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Acc = {}  NLL = {}  ({} samps)".format(acc/n_samp_seen,
                                                             nll/n_samp_seen,
                                                             n_samp_seen), end="")
    if verbosity > 0:
        print()
    nll = nll/(Xdev.shape[0]) ## calc full dev-set nll
    acc = acc/(Xdev.shape[0]) ## calc full dev-set acc
    return nll, acc

trAcc_set = []
trNll_set = []
acc_set = []
nll_set = []

sim_start_time = time.time() ## start time profiling

tr_acc = 0.1
nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
bestDevAcc = acc
print("-1: Dev: Acc = {}  NLL = {} | Tr: Acc = {}".format(acc, nll, tr_acc))
if verbosity >= 2:
    print(model._get_norm_string())
trAcc_set.append(tr_acc) ## random guessing is where models typically start
trNll_set.append(2.4)
acc_set.append(acc)
nll_set.append(nll)
jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/trNll.npy", jnp.asarray(trNll_set))
jnp.save("exp/nll.npy", jnp.asarray(nll_set))

for i in range(n_iter):
    ## shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]
    Y = _Y[ptrs,:]

    ## begin a single epoch/iteration
    n_samp_seen = 0
    tr_nll = 0.
    tr_acc = 0.
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)

        ## sample mini-batch of patterns
        idx = j * mb_size #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]
        Yb = Y[idx: idx + mb_size,:]
        ## perform a step of inference/learning
        _S, yMu, yCnt = model.process(obs=Xb, lab=Yb, adapt_synapses=True,label_dist_estimator=lab_estimator)
        ## track "online" training log likelihood and accuracy
        tr_nll += measure_CatNLL(yMu, Yb) * mb_size ## un-normalize score
        tr_acc += measure_ACC(yCnt, Yb) * mb_size ## un-normalize score
        n_samp_seen += Yb.shape[0]
        if verbosity >= 1:
            wStats = "" #model.get_synapse_stats()
            print("\r NLL = {} ACC = {} ({}) over {} samples ".format((tr_nll/n_samp_seen),
                                                                      (tr_acc/n_samp_seen),
                                                                      wStats, n_samp_seen), end="")
    if verbosity >= 1:
        print()

    ## evaluate current progress of model on dev-set
    nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
    tr_acc = (tr_acc/n_samp_seen)
    tr_nll = (tr_nll/n_samp_seen)
    if acc >= bestDevAcc:
        model.save_to_disk(params_only=True) # save final state of synapses to disk
        bestDevAcc = acc
    if (i+1) % save_point == 0 or i == (n_iter-1):
        jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
        jnp.save("exp/acc.npy", jnp.asarray(acc_set))
        jnp.save("exp/trNll.npy", jnp.asarray(trNll_set))
        jnp.save("exp/nll.npy", jnp.asarray(nll_set))
    ## record current generalization stats and print to I/O
    trAcc_set.append(tr_acc)
    acc_set.append(acc)
    trNll_set.append(tr_nll)
    nll_set.append(nll)
    io_str = ("{} Dev: Acc = {}, NLL = {} | "
              "Tr: Acc = {}, NLL = {}"
             ).format(i, acc, nll, tr_acc, tr_nll)
    if verbosity >= 1:
        print(io_str)
    else:
        print("\r{}".format(io_str), end="")
    if verbosity >= 2:
        print(model._get_norm_string())
if verbosity == 0:
    print("")

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

print("------------------------------------")
vAcc_best = jnp.amax(jnp.asarray(acc_set))
print(" Trial.sim_time = {} h  ({} sec)  Best Acc = {}".format(sim_time_hr, sim_time, vAcc_best))

jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/trNll.npy", jnp.asarray(trNll_set))
jnp.save("exp/nll.npy", jnp.asarray(nll_set))
