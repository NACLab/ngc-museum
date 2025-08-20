from jax import numpy as jnp, random, nn, jit
import numpy as np, time
import sys, getopt as gopt, optparse
## bring in ngc-learn analysis tools
from ngclearn.utils.viz.raster import create_raster_plot
import ngclearn.utils.metric_utils as metrics

import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet

def plot_clusters(W1, W2, figdim=50): ## generalized dual-layer plotting code
    n_input = W1.weights.value.shape[0] ## input layer
    n_hid1 = W1.weights.value.shape[1] ## layer size 1
    n_hid2 = W2.weights.value.shape[1] ## layer size 2
    ndim0 = int(jnp.sqrt(n_input)) ## sqrt(layer size 0)
    ndim1 = int(jnp.sqrt(n_hid1)) ## sqrt(layer size 1)
    ndim2 = int(jnp.sqrt(n_hid2)) ## sqrt(layer size 2)
    plt.figure(figsize=(figdim, figdim))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for q in range(W2.weights.value.shape[1]):
        masked = W1.weights.value * W2.weights.value[:, q]

        dim = ((ndim0 * ndim1) + (ndim1 - 1)) #(28 * 10) + (10 - 1)

        full = jnp.ones((dim, dim)) * jnp.amax(masked)

        for k in range(n_hid1):
            r = k // ndim1 #k // 10 #sqrt(hidden layer size)
            c = k % ndim1 # k % 10

            full = full.at[(r * (ndim0 + 1)):(r + 1) * ndim0 + r,
                           (c * (ndim0 + 1)):(c + 1) * ndim0 + c].set(
                jnp.reshape(masked[:, k], (ndim0, ndim0)))

        plt.subplot(ndim2, ndim2, q + 1) # 5 = sqrt(output layer size)
        plt.imshow(full, cmap=plt.cm.bone, interpolation='nearest')
        plt.axis("off")

    plt.subplots_adjust(top=0.9)
    plt.savefig("{}clusters.jpg".format(exp_dir), bbox_inches='tight')
    plt.clf()
    plt.close()

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["dataX=", "dataY=",
                                                    "model_dir=", "model_type=",
                                                    "label_fname=", "exp_dir=",
                                                    "param_subdir=",
                                                    "disable_adaptation=",
                                                    "make_cluster_plot="])

model_case = "snn_case1"
exp_dir = "exp/"
label_fname = "exp/labs.npy"
model_type = "tistdp"
model_dir = "exp/tistdp"
param_subdir = "/custom"
disable_adaptation = True
dataX = "../../data/mnist/trainX.npy"
dataY = "../../data/mnist/trainY.npy"
make_cluster_plot = True
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ('--model_dir'):
        model_dir = arg.strip()
    elif opt in ('--model_type'):
        model_type = arg.strip()
    elif opt in ('--label_fname'):
        label_fname = arg.strip()
    elif opt in ('--exp_dir'):
        exp_dir = arg.strip()
    elif opt in ('--param_subdir'):
        param_subdir = arg.strip()
    elif opt in ('--disable_adaptation'):
        disable_adaptation = (arg.strip().lower() == "true")
        print(" > Disable short-term adaptation? ", disable_adaptation)
    elif opt in ('--make_cluster_plot'):
        make_cluster_plot = (arg.strip().lower() == "true")
        print(" > Make cluster plot? ", make_cluster_plot)

if model_case == "snn_case1":
    print(" >> Setting up Case 1 model!")
    from snn_case1 import load_from_disk, get_nodes
elif model_case == "snn_case2":
    print(" >> Setting up Case 2 model!")
    from snn_case2 import load_from_disk, get_nodes
else:
    print("Error: No other model case studies supported! (", model_case, " invalid)")
    exit()

print(">> X: {}  Y: {}".format(dataX, dataY))

T = 250 # 300
dt = 1.

## load dataset
batch_size = 1 #100
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
n_batches = int(_X.shape[0]/batch_size)
patch_shape = (28, 28)

dkey = random.PRNGKey(time.time_ns())
model = load_from_disk(model_dir, param_dir=param_subdir,
                       disable_adaptation=disable_adaptation)
nodes = model.get_components("W1", "W1ie", "W1ei", "z0", "z1e", "z1i",
                             "W2", "W2ie", "W2ei", "z2e", "z2i")
W1, W1ie, W1ei, z0, z1e, z1i, W2, W2ie, W2ei, z2e, z2i = nodes

if make_cluster_plot:
    ## plot clusters formed by 2nd layer of model's spikes
    print(" >> Creating model cluster plot...")
    plot_clusters(W1, W2)

## extract label bindings
bindings = jnp.load(label_fname)

acc = 0.
Ns = 0.
yMu = []
for i in range(n_batches):
    Xb = _X[i * batch_size:(i+1) * batch_size, :]
    Yb = _Y[i * batch_size:(i+1) * batch_size, :]
    Ns += Xb.shape[0]
    model.reset()
    model.clamp(Xb)
    spikes1, spikes2 = model.infer(
        jnp.array([[dt * k, dt] for k in range(T)]))
    winner = jnp.argmax(jnp.sum(spikes2, axis=0))
    yHat = nn.one_hot(bindings[:, winner], num_classes=Yb.shape[1])
    yMu.append(yHat)
    acc = metrics.measure_ACC(yHat, Yb) + acc
    print("\r Acc = {} (over {} samples)".format(acc/Ns, i+1), end="")
print()
print("===============================================")
yMu = jnp.concatenate(yMu, axis=0)
conf_matrix, precision, recall, misses, acc, adj_acc = metrics.analyze_scores(yMu, _Y)
print(conf_matrix)
print("---")
print(" >> Number of Misses = {}".format(misses))
print(" >> Acc = {}  Precision = {}  Recall = {}".format(acc, precision, recall))
msg = "{}".format(conf_matrix)
jnp.save("{}confusion.npy".format(exp_dir), conf_matrix)
msg = ("{}\n---\n"
       "Number of Misses = {}\n"
       "Acc = {}  Adjusted-Acc = {}\n"
       "Precision = {}\nRecall = {}\n").format(msg, misses, acc, adj_acc, precision, recall)
fd = open("{}scores.txt".format(exp_dir), "w")
fd.write(msg)
fd.close()

