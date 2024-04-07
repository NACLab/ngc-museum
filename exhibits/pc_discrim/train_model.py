from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse
## bring in model from museum
from pc_model import PCN
## bring in ngc-learn analysis tools
from ngclearn.utils.model_utils import measure_ACC, measure_CatNLL

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY="]
                                 )
# GPU arguments
dataX = "../data/baby_mnist/babyX.npy"
dataY = "../data/baby_mnist/babyY.npy"
devX = dataX
devY = dataY
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
print("Data X: {} | Y: {}".format(dataX, dataY))

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_iter = 100
mb_size = 250
n_batches = int(_X.shape[0]/mb_size)

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

## build model
model = PCN(subkeys[1], x_dim, y_dim, hid1_dim=512, hid2_dim=512, T=20,
            dt=1., tau_m=20., act_fx="sigmoid", eta=0.001, exp_dir="exp",
            model_name="pcn")
model.save_to_disk() # save final state of synapses to disk

def eval_model(model, Xdev, Ydev, mb_size):
    n_batches = int(Xdev.shape[0]/mb_size)

    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    EFE = 0. ## free energy
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size,:]
        Yb = Ydev[idx: idx + mb_size,:]
        ## run model inference
        yMu_0, yMu, _EFE = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        ## record metric measurements
        _nll = measure_CatNLL(yMu_0, Yb) * Xb.shape[0] ## un-normalize score
        _acc = measure_ACC(yMu_0, Yb) * Yb.shape[0] ## un-normalize score
        nll += _nll
        acc += _acc
        EFE += _EFE

        n_samp_seen += Yb.shape[0]

    nll = nll/(Xdev.shape[0]) ## calc full dev-set nll
    acc = acc/(Xdev.shape[0]) ## calc full dev-set acc
    EFE = EFE/(Xdev.shape[0]) ## calc full dev-set EFE
    return nll, acc, EFE

nll_set = []
acc_set = []
efe_set = []

nll, acc, EFE = eval_model(model, Xdev, Ydev, mb_size=1000)
print("-1: Acc = {}  NLL = {}  EFE = {}".format(acc, nll, EFE))
#print(model.get_norm_string())
nll_set.append(nll)
acc_set.append(acc)
efe_set.append(EFE)
jnp.save("exp/nll.npy", jnp.asarray(nll_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/efe.npy", jnp.asarray(efe_set))

for i in range(n_iter):
    ## shuffle data (to ensure i.i.d. assumption holds)
    dkey, *subkeys = random.split(dkey, 2)
    ptrs = random.permutation(subkeys[0],_X.shape[0])
    X = _X[ptrs,:]
    Y = _Y[ptrs,:]

    ## begin a single epoch
    n_samp_seen = 0
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        ## sample mini-batch of patterns
        idx = j * mb_size #j % 2 # 1
        Xb = X[idx: idx + mb_size,:]
        Yb = Y[idx: idx + mb_size,:]
        ## perform a step of inference/learning
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        n_samp_seen += Yb.shape[0]
        print("\r {} processed ".format(n_samp_seen), end="")
    print()

    ## evaluate current progress of model on dev-set
    nll, acc, EFE = eval_model(model, Xdev, Ydev, mb_size=1000)
    model.save_to_disk() # save final state of synapses to disk
    nll_set.append(nll)
    acc_set.append(acc)
    efe_set.append(EFE)
    print("{}: Acc = {}  NLL = {}  EFE = {}".format(i, acc, nll, EFE))
    #print(model.get_norm_string())
    model.viz_receptive_fields(fname="recFields", field_shape=patch_shape,
                               show_stats=False)

jnp.save("exp/nll.npy", jnp.asarray(nll_set))
jnp.save("exp/acc.npy", jnp.asarray(acc_set))
jnp.save("exp/efe.npy", jnp.asarray(efe_set))
