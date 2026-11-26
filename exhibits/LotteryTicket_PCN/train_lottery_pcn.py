from jax import numpy as jnp, random
import sys, getopt as gopt, time
from lottery_pcn import LotteryPCN
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

"""
################################################################################
Lottery Ticket Predictive Coding Network (LotteryPCN) Exhibit File:

Fits a PCN classifier to the MNIST database over n_rounds; after each round, the
lowest-magnitude synapses are pruned and the active synapses are reinitialized
to their initial values (the backprop version is the lottery ticket hypothesis of
Frankle & Carbin, 2019).

Usage:
$ python train_lottery_pcn.py --dataX="/path/to/train_patterns.npy" \
                             --dataY="/path/to/train_labels.npy" \
                             --devX="/path/to/dev_patterns.npy" \
                             --devY="/path/to/dev_labels.npy" \

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=", "devX=", "devY="])
# external dataset arguments
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
devX = "../../data/mnist/validX.npy"
devY = "../../data/mnist/validY.npy"
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))

exp_dir = "exp"
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_iter = 10           ## number of epochs per round
n_rounds = 15         ## number of (train > prune > reset_weights) rounds
prune_rate = 0.2     ## ratio of the pruned weights in each round
reinit = False       ## if True, active synapses get new random values instead of their initial ones

save_point = 1       ## save model params every modulo "save_point"
mb_size = 250
n_batches = int(_X.shape[0] / mb_size)
## set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build model
print("--- Building Model ---")
model = LotteryPCN(model_name="Lottery_PCN", dkey=subkeys[1],
                   in_dim=x_dim,
                   out_dim=y_dim,
                   hid1_dim=100,
                   hid2_dim=64,
                   T=20,
                   dt=1.,
                   tau_m=25.,
                   act_fx="relu",
                   eta=0.001,
                   weight_init=dist.fan_in_gaussian(gain=jnp.sqrt(2.)),
                   exp_dir=exp_dir,
)
model.save_to_disk()  # save final state of synapses to disk
# model.load_from_disk(exp_dir)
print("--- Starting Simulation ---")

def eval_model(model, Xdev, Ydev, mb_size):  ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0] / mb_size)

    n_samp_seen = 0
    nll = 0.  ## negative Categorical log likelihood
    acc = 0.  ## accuracy
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]
        ## run model inference
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        ## record metric measurements
        _nll = measure_CatNLL(yMu_0, Yb) * Xb.shape[0]  ## un-normalize score
        _acc = measure_ACC(yMu_0, Yb) * Yb.shape[0]  ## un-normalize score
        nll += _nll
        acc += _acc

        n_samp_seen += Yb.shape[0]

    nll = nll / (Xdev.shape[0])  ## calc full dev-set nll
    acc = acc / (Xdev.shape[0])  ## calc full dev-set acc
    return nll, acc

trAcc_set = []
acc_set = []
efe_set = []
nll_set = []
ticket_set = []         ## per round: synapses kept (%)

sim_start_time = time.time()  ## start time profiling

_, tr_acc = eval_model(model, _X, _Y, mb_size=1000)
nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
print("-1: Dev: Acc = {:.2f}  NLL = {:.3f} | Tr: Acc = {:.2f} EFE = --".format(acc, nll, tr_acc))
print(model._get_norm_string())
trAcc_set.append(tr_acc)  ## random guessing is where models typically start
acc_set.append(acc)
nll_set.append(nll)
efe_set.append(-2000.)
jnp.save(exp_dir + "/acc.npy", jnp.asarray(acc_set))
jnp.save(exp_dir + "/efe.npy", jnp.asarray(efe_set))

epoch_times = []
kept = 100.  ## percent of synapses kept starts with full network
for i in range(n_rounds * n_iter):
    ## start of a new round: prune the trained synapses, then reset_weights of the active ones
    if i > 0 and i % n_iter == 0:
        dkey, *subkeys = random.split(dkey, 2)
        model.prune(prune_rate)
        model.reset_weights(subkeys[0] if reinit else None)
        kept = 100. * float(jnp.mean(jnp.concatenate([model.W1.weights.get().ravel(),
                                                      model.W2.weights.get().ravel(),
                                                      model.W3.weights.get().ravel()]) != 0.))

    ## shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    ###############################################################################
    ## begin a single epoch (includes pass thru data + model evals calls)
    n_samp_seen = 0
    train_EFE = 0.  ## training free energy (online) estimate
    trAcc = 0.  ## training accuracy score
    epoch_time = time.time()
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)

        ## sample mini-batch of patterns
        idx = j * mb_size  # j % 2 # 1
        Xb = X[idx: idx + mb_size, :]
        Yb = Y[idx: idx + mb_size, :]

        ## perform a step of inference/learning
        yMu_0, yMu, _EFE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)

        ## track online training EFE and accuracy
        train_EFE += _EFE * mb_size
        n_samp_seen += Yb.shape[0]
        print("\r Iter ({}/{}) EFE = {:.3f} over {}/{} samples | kept = {:.1f}%       ".format(
            i, n_iter, (train_EFE / n_samp_seen), n_samp_seen, len(Y), kept), end=""
            )

    ## evaluate current progress of model on dev-set
    nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
    _, tr_acc = eval_model(model, _X, _Y, mb_size=1000)

    epoch_time = time.time() - epoch_time
    epoch_times.append(epoch_time)
    ###############################################################################

    if (i + 1) % save_point == 0 or i == (n_rounds * n_iter - 1):
        model.save_to_disk(params_only=True)  # save final state of synapses to disk
        jnp.save(exp_dir + "/trAcc.npy", jnp.asarray(trAcc_set))
        jnp.save(exp_dir + "/acc.npy", jnp.asarray(acc_set))
        jnp.save(exp_dir + "/efe.npy", jnp.asarray(efe_set))

    ## record current generalization stats and print to I/O
    trAcc_set.append(tr_acc)
    acc_set.append(acc)
    nll_set.append(nll)
    efe_set.append((train_EFE / n_samp_seen))
    print(f"(Round {i // n_iter}: Iter {i % n_iter + 1}/{n_iter}) "
              f"| Dev: Acc = {acc:.2f}, NLL = {nll:.3f} "
              f"| Tr: Acc = {tr_acc:.2f}, EFE = {train_EFE / n_samp_seen:.3f} "
              )

    ## end of a round: get the early-stopping epoch (minimum dev NLL) and the accuracy at that epoch
    if (i + 1) % n_iter == 0:
        stop = int(jnp.argmin(jnp.asarray(nll_set[-n_iter:])))
        ticket_set.append((kept, stop + 1, acc_set[-n_iter:][stop]))
        print("\n >> Round {} | kept = {:.1f}% | Early-stop epoch = {} | Dev: Acc = {:.4f}".format(
            i // n_iter, kept, stop + 1, acc_set[-n_iter:][stop]))

        makedir("{}/round{}".format(model.exp_dir +"/"+ model.model_name, i // n_iter))  ## keep the trained synapses of this round
        for W in (model.W1, model.W2, model.W3):
            W.save("{}/round{}".format(model.exp_dir +"/"+ model.model_name, i // n_iter))
        jnp.save("{}/{}.npy".format(model.exp_dir, "reinit" if reinit else "ticket"), jnp.asarray(ticket_set))
print("")

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time / 3600.0)          ## convert time to hours

print("------------------------------------")
vAcc_best = jnp.amax(jnp.asarray(acc_set))
epoch_time_mu = jnp.mean(jnp.array(epoch_times))
epoch_time_sig = jnp.std(jnp.array(epoch_times))
print(
    f" Trial.sim_time = {sim_time_hr:.3f} h ({sim_time:.2f} sec); "
    f"Epoch.time = {epoch_time_mu:.2f} +- {epoch_time_sig:.2f}; "
    f"Best Acc = {vAcc_best:.2f}"
)

jnp.save(exp_dir + "/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save(exp_dir + "/acc.npy", jnp.asarray(acc_set))
jnp.save(exp_dir + "/efe.npy", jnp.asarray(efe_set))






