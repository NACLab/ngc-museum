from jax import numpy as jnp, random
import sys, getopt as gopt, time
from pcn_vit import PCN_ViT
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn.components.input_encoders.ganglionCell import _create_patches
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

"""
################################################################################
Predictive Coding Network Vision Transformer (PCN_ViT) Exhibit File:

Fits a PCN classifier to the MNIST database whose images are cut into non-overlapping
patches (tokens); the first synapse embeds each patch and the second synapse lets the
patches attend to one another (the backprop version is the ViT of Dosovitskiy et al., 2021).

Usage:
$ python train_pcn_vit.py --dataX="/path/to/train_patterns.npy" \
                            --dataY="/path/to/train_labels.npy" \
                            --devX="/path/to/dev_patterns.npy" \
                            --devY="/path/to/dev_labels.npy"

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY="]
                                 )
data_path = "../../data/mnist"

# external dataset arguments
dataX = data_path + "/trainX.npy"
dataY = data_path + "/trainY.npy"
devX = data_path + "/validX.npy"
devY = data_path + "/validY.npy"

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

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
token_dim = 64

ix, iy = image_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
px, py = patch_shape = (7, 7)
n_patch = (ix // px) * (iy // py)
y_dim = _Y.shape[1]

## patching into non-overlapping patches (tokens)
_X = _create_patches(_X.reshape(-1, ix, iy), patch_shape, patch_shape).reshape(-1, x_dim)
Xdev = _create_patches(Xdev.reshape(-1, ix, iy), patch_shape, patch_shape).reshape(-1, x_dim)

n_iter = 20
mb_size = 250
n_batches = int(_X.shape[0]/mb_size)
save_point = 1 ## save model params every modulo "save_point"
mb_vis_size = 100  ## number of W1 receptive fields (filters) to visualize per patch (-1 = all)

## set up JAX seeding
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build model
print("--- Building Model ---")
model = PCN_ViT(model_name="pcn_vit",
                use_value=True,
                in_dim=x_dim,
                out_dim=y_dim,
                hid1_dim=n_patch * token_dim,
                hid2_dim=n_patch * token_dim,
                T=20,
                dt=1.,
                tau_m=25.,
                n_patch=n_patch,
                image_shape=image_shape,
                patch_shape=patch_shape,
                act_fx="tanh",
                eta=0.002,
                weight_init=dist.fan_in_gaussian(),
                exp_dir="exp",
                dkey=subkeys[1]
                )
model.save_to_disk() # save final state of synapses to disk
# model.load_from_disk("exp")

def eval_model(model, Xdev, Ydev, mb_size): ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/mb_size)

    n_samp_seen = 0
    nll = 0. ## negative Categorical log likelihood
    acc = 0. ## accuracy
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size,:]
        Yb = Ydev[idx: idx + mb_size,:]
        ## run model inference
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        ## record metric measurements
        _nll = measure_CatNLL(yMu_0, Yb) * Xb.shape[0] ## un-normalize score
        _acc = measure_ACC(yMu_0, Yb) * Yb.shape[0] ## un-normalize score
        nll += _nll
        acc += _acc

        n_samp_seen += Yb.shape[0]

    nll = nll/(Xdev.shape[0]) ## calc full dev-set nll
    acc = acc/(Xdev.shape[0]) ## calc full dev-set acc
    return nll, acc

trAcc_set = []
acc_set = []
efe_set = []

sim_start_time = time.time() ## start time profiling

_, tr_acc = eval_model(model, _X, _Y, mb_size=1000)
nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
print("-1:  | Tr: Acc = {:.2f}. | Dev: Acc = {:.2f}  NLL = {:.3f}".format(tr_acc, acc, nll))

trAcc_set.append(tr_acc) ## random guessing is where models typically start
acc_set.append(acc)
efe_set.append(-2000.)
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/efe.npy", jnp.asarray(efe_set))

epoch_times = []
for i in range(n_iter):
    ## shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]
    Y = _Y[ptrs,:]

    ###############################################################################
    ## begin a single epoch (includes pass thru data + model evals calls)
    n_samp_seen = 0
    train_EFE = 0. ## training free energy (online) estimate
    trAcc = 0. ## training accuracy score
    epoch_time = time.time()
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        
        ## sample mini-batch of patterns
        idx = j * mb_size #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]
        Yb = Y[idx: idx + mb_size,:]
        
        ## perform a step of inference/learning
        yMu_0, yMu, _EFE = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        
        ## track online training EFE and accuracy   (summed over the batch)
        train_EFE += _EFE
        n_samp_seen += Yb.shape[0]
        print("\r EFE = {:.3f} over {} samples ".format((train_EFE / n_samp_seen), n_samp_seen), end="")
    
    ## evaluate current progress of model on dev-set
    nll, acc = eval_model(model, Xdev, Ydev, mb_size=1000)
    _, tr_acc = eval_model(model, _X, _Y, mb_size=1000)
    
    epoch_time = time.time() - epoch_time
    epoch_times.append(epoch_time)

    ## record current generalization stats and print to I/O
    trAcc_set.append(tr_acc)
    acc_set.append(acc)
    efe_set.append((train_EFE / n_samp_seen))
    ###############################################################################

    if (i+1) % save_point == 0 or i == (n_iter-1):
        model.save_to_disk(params_only=True) # save final state of synapses to disk
        jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
        jnp.save("exp/acc.npy", jnp.asarray(acc_set))
        jnp.save("exp/efe.npy", jnp.asarray(efe_set))

    print("\r"
          f"({i + 1}/{n_iter}) "
          f"| Train-EFE = {train_EFE / n_samp_seen:.3f} "
          f"| Train Acc = {tr_acc:.2f} "
          f"| Dev: Acc = {acc:.2f}"
          )

## stop time profiling
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time
sim_time_hr = (sim_time/3600.0) # convert time to hours

print("------------------------------------")
vAcc_best = jnp.amax(jnp.asarray(acc_set))
epoch_time_mu = jnp.mean(jnp.array(epoch_times))
epoch_time_sig = jnp.std(jnp.array(epoch_times))
print(
    f" Trial.sim_time = {sim_time_hr:.3f} h ({sim_time:.2f} sec); "
    f"Epoch.time = {epoch_time_mu:.2f} +- {epoch_time_sig:.2f}; "
    f"Best Acc = {vAcc_best:.2f}"
)

jnp.save("exp/trAcc.npy", jnp.asarray(trAcc_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/efe.npy", jnp.asarray(efe_set))



