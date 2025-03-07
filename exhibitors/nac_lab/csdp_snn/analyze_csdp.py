from jax import numpy as jnp, random, nn, jit
import numpy as np
import sys, getopt as gopt, optparse
from csdp_model import CSDP_SNN as Model
from ngclearn.utils.io_utils import makedir
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, extract_pca_latents, plot_latents

"""
################################################################################
CSDP Exhibit File:

Evaluates CSDP on the MNIST database.

Usage:
$ python analyze_csdp.py --dataX="/path/to/data_patterns.npy" \
                         --dataY="/path/to/labels.npy" \
                         --verbosity=0 \
                         --modelDir=exp/

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

def measure_BCE(p, x, offset=1e-7, preserve_batch=False):
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 3)

########################################################################
## declare a program-specific test-time evaluation co-routine
def eval_model(model, Xdev, Ydev, mb_size, verbosity=1,
               concat_all_layer_codes=False): ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/mb_size)

    latents = []
    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    bce = 0.
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * mb_size
        Xb = Xdev[idx: idx + mb_size, :]
        Yb = Ydev[idx: idx + mb_size, :]
        ## run model inference
        yMu, yCnt, _R1, _R2, _R3, xMu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=False, collect_rate_codes=True,
            lab_estimator="softmax", collect_recon=collect_recon
        )
        ## record metric measurements
        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0] ## un-normalize score
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0] ## un-normalize score
        _bce = measure_BCE(xMu, Xb) * Xb.shape[0] ## un-normalize score
        nll += _nll
        acc += _acc
        bce += _bce
        if concat_all_layer_codes:
            latents.append(jnp.concatenate((_R1, _R2), axis=1))
        else:
            latents.append(_R2)

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Acc = {}  NLL = {}  BCE = {} ({} samps)".format(acc/n_samp_seen,
                                                                    nll/n_samp_seen,
                                                                    bce/n_samp_seen,
                                                                    n_samp_seen), end="")
    if verbosity > 0:
        print()
    nll = nll/(Xdev.shape[0]) ## calc full dev-set nll
    acc = acc/(Xdev.shape[0]) ## calc full dev-set acc
    latents = jnp.concatenate(latents, axis=0)
    return nll, acc, latents
########################################################################

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=", "verbosity=",
                                                    "modelDir=", "paramDir=",
                                                    "codebookName=", "viz_tsne="])

viz_tsne = True
modelDir = "exp/"
paramDir = "/best_params1234"
codebookName = "codes"
dataX = "../data/mnist/validX.npy"
dataY = "../data/mnist/validY.npy"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--modelDir"):
        modelDir = arg.strip()
    elif opt in ("--paramDir"):
        paramDir = "/{}".format(arg.strip())
    elif opt in ("--codebookName"):
        codebookName = arg.strip()
    elif opt in ("--viz_tsne"):
        viz_tsne = (arg.strip().lower() == "true")
print("=> Data X: {} | Y: {}".format(dataX, dataY))

## load dataset
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
n_batches = _X.shape[0]
patch_shape = (28, 28)

dkey = random.PRNGKey(1234)

collect_recon = True
algo_type = "supervised"
T = 50
dt = 3.
concat_all_layer_codes = False #True

## load in/reconstruct saved model from disk
model = Model(dkey, T=T, dt=dt, load_model_dir="{}/snn_csdp".format(modelDir),
              load_param_subdir=paramDir)

## evaluate performance
nll, acc, latents = eval_model(model, _X, _Y, mb_size=1000, verbosity=verbosity, 
                               concat_all_layer_codes=concat_all_layer_codes)
print("------------------------------------")
print("=> Test.NLL = {:.5f}  Acc = {:.3f}".format(nll, acc * 100.))

codes_fname = "{}{}.npy".format(modelDir, codebookName)
print(" >> Saving model codes: ", codes_fname, " Shape: ", latents.shape)
jnp.save(codes_fname, latents)

## extract latents and visualize via the t-SNE algorithm
if viz_tsne:
    print(" >> Constructing tSNE visualization of model (rate-)codes...")
    tsneDir = "{}/tsne/".format(modelDir)
    codes_fname = "{}{}_tsne.npy".format(tsneDir, codebookName)
    makedir(tsneDir)

    print("rate-code latents.shape = ", latents.shape)
    codes = extract_tsne_latents(np.asarray(latents), perplexity=30, n_pca_comp=400)
    print("tSNE-codes.shape = ", codes.shape)
    jnp.save(codes_fname, codes) ## save tSNE codes to disk
    
    ## produce tSNE plot of final projected codes
    alpha = 1. # set to value > 0 and < 1 if more transparent clusters are desired
    plot_latents(codes, _Y, plot_fname="{}snn_tsne_codes.png".format(tsneDir), alpha=alpha)
