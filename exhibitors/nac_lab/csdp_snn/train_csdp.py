from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time

from csdp_model import CSDP_SNN as Model
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

"""
################################################################################
CSDP-Trained Spiking Neural Network (CSDP-SNN) Exhibit File:

Fits a CSDP-SNN (an SNN trained with contrastive-signal-dependent plasticity)
classifier to a database of patterns (with labels).

Usage:
$ python train_csdp.py --dataX="/path/to/train_patterns.npy" \
                         --dataY="/path/to/train_labels.npy" \
                         --devX="/path/to/dev_patterns.npy" \
                         --devY="/path/to/dev_labels.npy" \
                         --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

################################################################################
## declare some useful co-routines for measuring performance

def measure_BCE(p, x, offset=1e-7, preserve_batch=False): ## binary cross-entropy
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce

def measure_MSE(mu, x, preserve_batch=False): ## mean squared error
    diff = mu - x
    se = jnp.square(diff) ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True) # technically se at this point
    if preserve_batch is False:
        mse = jnp.mean(mse) # this is proper mse
    return mse
################################################################################

# read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '',
    ["dataX=", "dataY=", "devX=", "devY=", "algo_type=", "num_iter=",
     "verbosity=", "seed=", "exp_dir=", "nZ1=", "nZ2="]
)
# external dataset arguments
nZ1 = 1024
nZ2 = 128
algo_type = "supervised" #"unsupervised"
seed = 1234
num_iter = 30 ## epochs
batch_size = 500
dev_batch_size = 1000
dataX = "data/mnist/trainX.npy"
dataY = "data/mnist/trainY.npy"
devX = "data/mnist/validX.npy"
devY = "data/mnist/validY.npy"
exp_dir = "exp"
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
    elif opt in ("--algo_type"):
        algo_type = arg.strip()
    elif opt in ("--num_iter"):
        num_iter = int(arg.strip())
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--nZ1"):
        nZ1 = int(arg.strip())
    elif opt in ("--nZ2"):
        nZ2 = int(arg.strip())
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()
print("#####################################################")
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))
print("#####################################################")

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_batches = int(_X.shape[0]/batch_size) ## get number batches (for progress reporting)
save_point = 5 ## save model params every epoch/iteration modulo "save_point"

## set up JAX seeding
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 10)

########################################################################
## configure model
hid_dim = nZ1
hid_dim2 = nZ2
out_dim = y_dim ## output dimensionality
learn_recon = True 
T = 50
eta_w = 0.002  ## learning rate -- CSDP/hebbian update modulation
dt = 3. # ms ## integration time constant
model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, hid_dim2=hid_dim2,
              batch_size=batch_size, eta_w=eta_w, T=T, dt=dt, algo_type=algo_type,
              exp_dir=exp_dir, learn_recon=learn_recon)
model.save_to_disk(save_dir="custom{}".format(seed)) # save final state of synapses to disk
########################################################################

########################################################################
## declare a simple test-time evaluation co-routine
def eval_model(model, Xdev, Ydev, batch_size, verbosity=1):
    ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/batch_size)

    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    bce = 0. ## bin cross-entropy
    mse = 0. ## mean-squared error
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * batch_size
        Xb = Xdev[idx: idx + batch_size, :]
        Yb = Ydev[idx: idx + batch_size, :]

        ## run model inference
        yMu, yCnt, _, _, _, xMu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=False)
        ## record metric measurements (note: must also un-normalizing them here)
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0]
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0] 
        _bce = measure_BCE(xMu, Xb, preserve_batch=False) * Xb.shape[0]
        _mse = measure_MSE(xMu, Xb, preserve_batch=False) * Xb.shape[0]
        nll += _nll
        acc += _acc
        bce += _bce
        mse += _mse

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Eval.Step:  Acc = {:.3f}; NLL = {:.5f}; CE = {:.5f}; MSE = {:.5f} ".format(
                (acc/n_samp_seen) * 100., nll/n_samp_seen, bce/n_samp_seen,
                mse/n_samp_seen), end=""
            )
    if verbosity > 0:
        print()
    ## produce final measurements
    nll = nll/(Xdev.shape[0])
    acc = acc/(Xdev.shape[0])
    bce = bce/(Xdev.shape[0])
    mse = mse/(Xdev.shape[0])
    return nll, acc, bce, mse
########################################################################

trAcc_set = []
trNll_set = []
acc_set = []
nll_set = []
bce_set = []
mse_set = []

sim_start_time = time.time() ## start time profiling

tr_acc = 0.1
nll, acc, bce, mse = eval_model(
    model, Xdev, Ydev, batch_size=dev_batch_size, verbosity=verbosity
)
bestDevAcc = acc
if verbosity > 0:
    print(
        "########\n-1: V: Acc = {:.3f}, NLL = {:.5f} \n    BCE = {:.5f} MSE = {:.5f} | "
        "Tr: Acc = {:.3f}\n#######".format(acc * 100., nll, bce, mse, tr_acc)
    )
else:
    print(" Epoch -1 (Before training) ")
    print(
        "     Dev.NLL = {:.5f}  ACC = {:.3f}  BCE = {:.5f}".format(
            nll, acc * 100., bce)
    )


trAcc_set.append(tr_acc) ## random guessing is where models typically start
trNll_set.append(2.4)
acc_set.append(acc)
nll_set.append(nll)
bce_set.append(bce)
mse_set.append(mse)
jnp.save("{}/trAcc{}.npy".format(exp_dir, seed), jnp.asarray(trAcc_set))
jnp.save("{}/acc{}.npy".format(exp_dir, seed), jnp.asarray(acc_set))
jnp.save("{}/trNll{}.npy".format(exp_dir, seed), jnp.asarray(trNll_set))
jnp.save("{}/nll{}.npy".format(exp_dir, seed), jnp.asarray(nll_set))
jnp.save("{}/bce{}.npy".format(exp_dir, seed), jnp.asarray(bce_set))
jnp.save("{}/mse{}.npy".format(exp_dir, seed), jnp.asarray(mse_set))

@jit
def measure_acc_nll(yMu, Yb): ## this is just a fast compound accuracy/NLL function
    mask = jnp.concatenate((jnp.ones((Yb.shape[0],1)),jnp.zeros((Yb.shape[0],1))), axis=0)
    N = jnp.sum(mask)
    _Yb = jnp.concatenate((Yb,Yb), axis=0) * mask
    offset = 1e-6
    _yMu = jnp.clip(yMu * mask, offset, 1.0 - offset)
    loss = -(_yMu * jnp.log(_yMu))
    nll = jnp.sum(jnp.sum(loss, axis=1, keepdims=True) * mask) * (1./N)

    guess = jnp.argmax(yMu, axis=1, keepdims=True)
    lab = jnp.argmax(_Yb, axis=1, keepdims=True)
    acc = jnp.sum( jnp.equal(guess, lab) * mask )/(N)
    return acc, nll

############################ simulate training process #########################
for i in range(num_iter):
    ## shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    ## begin a single epoch/iteration
    n_samp_seen = 0
    tr_nll = 0.
    tr_acc = 0.
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        ## sample mini-batch of patterns
        idx = j * batch_size 
        s_ptr = idx
        e_ptr = idx + batch_size
        if e_ptr > X.shape[0]:
            e_ptr = X.shape[0]
        Xb = X[s_ptr: e_ptr, :]
        Yb = Y[s_ptr: e_ptr, :]

        ## perform a step of inference/learning
        yMu, yCnt, _, _, _, x_mu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=True, collect_rate_codes=True
        )
        ## track "online" training log likelihood and accuracy
        _tr_acc, _tr_nll = measure_acc_nll(yMu, Yb) # compute masked scores
        tr_nll += _tr_nll * Yb.shape[0] ## un-normalize score
        tr_acc += _tr_acc * Yb.shape[0] ## un-normalize score
        n_samp_seen += Yb.shape[0]

        train_msg = "\r Epoch {} (batch {}/{}): Train.NLL = {:.5f} ACC = {:.3f}".format(
            i, (j+1), n_batches, (tr_nll / n_samp_seen), (tr_acc / n_samp_seen) * 100.
        )
        if verbosity > 0:
            train_msg = "{} (Online training estimate)".format(train_msg)
        print(
            train_msg, end=""
        )
    print()

    ## evaluate current progress of model on dev-set
    nll, acc, bce, mse = eval_model(
        model, Xdev, Ydev, batch_size=dev_batch_size, verbosity=verbosity
    )
    if verbosity == 0:
        print("     Dev.NLL = {:.5f}  ACC = {:.3f}  BCE = {:.5f}".format(nll, acc * 100., bce))
    ## else, verbosity > 0 triggers the messier, richer print-out message below

    ## prepare and save biased, online training statistics to disk
    tr_acc = (tr_acc/n_samp_seen)
    tr_nll = (tr_nll/n_samp_seen)
    if acc >= bestDevAcc:
        model.save_to_disk(save_dir="best_params{}".format(seed), params_only=True) # save final state of synapses to disk
        bestDevAcc = acc
    if (i+1) % save_point == 0 or i == (num_iter-1):
        model.save_to_disk(save_dir="custom{}".format(seed), params_only=True)
    ## record current generalization stats and print to I/O
    trAcc_set.append(tr_acc)
    acc_set.append(acc)
    trNll_set.append(tr_nll)
    nll_set.append(nll)
    bce_set.append(bce)
    mse_set.append(mse)

    io_str = (
        "########\n{} V: Acc = {}, NLL = {}\n    BCE = {} MSE = {} | "
        "Tr: Acc = {}, NLL = {} \n########").format(
        i, acc, nll, bce, mse, tr_acc, tr_nll
    )
    if verbosity >= 1:
        print(io_str)
    # else:
    #     print("\r{}".format(io_str), end="")

    ## save current state of score arrays
    jnp.save("{}/trAcc{}.npy".format(exp_dir, seed), jnp.asarray(trAcc_set))
    jnp.save("{}/acc{}.npy".format(exp_dir, seed), jnp.asarray(acc_set))
    jnp.save("{}/trNll{}.npy".format(exp_dir, seed), jnp.asarray(trNll_set))
    jnp.save("{}/nll{}.npy".format(exp_dir, seed), jnp.asarray(nll_set))
    jnp.save("{}/bce{}.npy".format(exp_dir, seed), jnp.asarray(bce_set))
    jnp.save("{}/mse{}.npy".format(exp_dir, seed), jnp.asarray(mse_set))
########################################################################
if verbosity == 0:
    print("")

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

## print out a few final simulation stats to the user
print("------------------------------------")
vAcc_best = jnp.amax(jnp.asarray(acc_set))
print(" Trial.sim_time = {} h  ({} sec)  Best Acc = {}".format(sim_time_hr, sim_time, vAcc_best))

log = open("{}/sim_stats_{}.txt".format(exp_dir, seed), 'w')
log.write("Sim time: {} hr  {} sec".format(sim_time_hr, sim_time))
log.close() # close file descriptor

## save final simulation measurements to disk
jnp.save("{}/trAcc{}.npy".format(exp_dir,seed), jnp.asarray(trAcc_set))
jnp.save("{}/acc{}.npy".format(exp_dir,seed), jnp.asarray(acc_set))
jnp.save("{}/trNll{}.npy".format(exp_dir,seed), jnp.asarray(trNll_set))
jnp.save("{}/nll{}.npy".format(exp_dir,seed), jnp.asarray(nll_set))
jnp.save("{}/bce{}.npy".format(exp_dir,seed), jnp.asarray(bce_set))
jnp.save("{}/mse{}.npy".format(exp_dir,seed), jnp.asarray(mse_set))
