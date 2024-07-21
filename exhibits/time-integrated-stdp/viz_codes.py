from jax import numpy as jnp
import sys, getopt as gopt, optparse
from ngclearn.utils.viz.dim_reduce import extract_tsne_latents, plot_latents


# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["labels_fname=", "codes_fname=",
                                                    "plot_fname="])

plot_fname = "codes.jpg"
labels_fname = "../../data/mnist/testY.npy"
codes_fname = "exp/test_codes.npy"
for opt, arg in options:
    if opt in ("--labels_fname"):
        labels_fname = arg.strip()
    elif opt in ("--codes_fname"):
        codes_fname = arg.strip()
    elif opt in ("--plot_fname"):
        plot_fname = arg.strip()

labels = jnp.load(labels_fname)
print("Lab.shape: ", labels.shape)
codes = jnp.load(codes_fname)
print("Codes.shape: ", codes.shape)

## visualize the above data via the t-SNE algorithm
tsne_codes = extract_tsne_latents(codes)
print("tSNE-codes.shape = ", tsne_codes.shape)
plot_latents(tsne_codes, labels, plot_fname=plot_fname)