from jax import numpy as jnp, random, nn, jit
import numpy as np, time
import sys, getopt as gopt, optparse
## bring in ngc-learn analysis tools
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.metric_utils import measure_ACC

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=",
                                                    "codebook_fname=",
                                                    "model_fname=",
                                                    "n_samples=", "model_type=",
                                                    "param_subdir=",
                                                    "disable_adaptation="])

model_case = "snn_case1"
n_samples = -1
model_type = "tistdp"
model_fname = "exp/tistdp"
param_subdir = "/custom"
disable_adaptation = True
codebook_fname = "codes.npy"
dataX = "../../data/mnist/trainX.npy"
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--codebook_fname"):
        codebook_fname = arg.strip()
    elif opt in ("--model_fname"):
        model_fname = arg.strip()
    elif opt in ("--n_samples"):
        n_samples = int(arg.strip())
    elif opt in ('--model_type'):
        model_type = arg.strip()
    elif opt in ('--param_subdir'):
        param_subdir = arg.strip()
    elif opt in ('--disable_adaptation'):
        disable_adaptation = (arg.strip().lower() == "true")
        print(" >  Disable short-term adaptation? ", disable_adaptation)

if model_case == "snn_case1":
    print(" >> Setting up Case 1 model!")
    from snn_case1 import load_from_disk, get_nodes
elif model_case == "snn_case2":
    print(" >> Setting up Case 2 model!")
    from snn_case2 import load_from_disk, get_nodes
else:
    print("Error: No other model case studies supported! (", model_case, " invalid)")
    exit()

print(">> X: {}".format(dataX))

## load dataset
batch_size = 1 #100
_X = jnp.load(dataX)
if 0 < n_samples < _X.shape[0]:
    _X = _X[0:n_samples, :]
n_batches = int(_X.shape[0]/batch_size)
patch_shape = (28, 28)

dkey = random.PRNGKey(time.time_ns())
model = load_from_disk(model_directory=model_fname, param_dir=param_subdir,
                       disable_adaptation=disable_adaptation)

T = 250 # 300
dt = 1.

codes = [] ## get latent (spike) codes
acc = 0.
Ns = 0.
for i in range(n_batches):
    Xb = _X[i * batch_size:(i + 1) * batch_size, :]
    Ns += Xb.shape[0]
    model.reset()
    model.clamp(Xb)
    spikes1, spikes2 = model.infer(
        jnp.array([[dt * k, dt] for k in range(T)]))
    counts = jnp.sum(spikes2, axis=0) ## get counts
    codes.append(counts)
    print("\r > Processed ({} samples)".format(Ns), end="")
print()

print(" >> Saving code-book to disk: {}".format(codebook_fname))
codes = jnp.concatenate(codes, axis=0)
print(" >> Code.shape = ", codes.shape)
jnp.save(codebook_fname, codes)
