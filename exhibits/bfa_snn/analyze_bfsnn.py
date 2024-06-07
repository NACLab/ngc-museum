from jax import numpy as jnp, random, nn, jit
import numpy as np
import sys, getopt as gopt, optparse
from bfasnn_model import load_model
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, plot_latents
from bfasnn_model import BFA_SNN as Model ## bring in model from museum

"""
################################################################################
Broadcast Feedback Alignment Spiking Network (BFA-SNN) Exhibit File:

Evaluates a trained BFA-SNN classifier on the MNIST database test-set and produces
a t-SNE visualization of its penultimate neuronal layer activities over this
test-set.

Usage:
$ python analyze_bfsnn.py --dataX="/path/to/data_patterns.npy" \
                          --dataY="/path/to/labels.npy" \
                          --verbosity=0

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

## -------------------------------------------------
## some label prediction hyper-parameters
## (these below used to recover Samadi et al., 2017)
lab_estimator = "current"
#lab_estimator = "spikes"
acc_type = "spikes"
#acc_type = "soft"
## -------------------------------------------------

## program-specific co-routine
def eval_model(model, Xdev, Ydev, mb_size, verbosity=1):
    ## evals model's test-time inference performance and collect latent codes
    n_batches = int(Xdev.shape[0]/mb_size)

    latents = []
    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size,:]
        Yb = Ydev[idx: idx + mb_size,:]
        ## run model inference
        latent_rates, yMu, yCnt = model.process(obs=Xb, lab=Yb, adapt_synapses=False,
                                                label_dist_estimator=lab_estimator,
                                                get_latent_rates=True)
        latents.append(latent_rates)
        ## record metric measurements
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0] ## un-normalize score
        if acc_type == "spikes":
            _acc = measure_ACC(yCnt, Yb) * Yb.shape[0] ## un-normalize score
        else:
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
    latents = jnp.concatenate(latents, axis=0)
    return nll, acc, latents


# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=",
                                                    "verbosity="])

sample_idx = 0 ## choose a pattern (0 <= idx < _X.shape[0])
dataX = "../../data/mnist/validX.npy"
dataY= "../../data/mnist/validY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
print("=> Data X: {} | Y: {}".format(dataX, dataY))

## load dataset
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
n_batches = _X.shape[0]
patch_shape = (28, 28)

T = 100 # 80
dt = 0.25 #1.
dkey = random.PRNGKey(1234)
model = Model(dkey=dkey, dt=dt, T=T, loadDir="exp/snn_bfa")
#model = load_model("exp/snn_bfa", dt=dt, T=T) ## load in pre-trained SNN model

## evaluate performance
nll, acc, latents = eval_model(model, _X, _Y, mb_size=1000)
print("------------------------------------")
print("=> NLL = {}  Acc = {}".format(nll, acc))

## extract latents and visualize via the t-SNE algorithm
print("latent.shape = ",latents.shape)
codes = extract_tsne_latents(np.asarray(latents))
print("code.shape = ",codes.shape)
plot_latents(codes, _Y, plot_fname="exp/snn_codes.jpg")
