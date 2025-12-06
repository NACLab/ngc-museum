import sys, getopt as gopt, optparse, time
from jax import numpy as jnp, random
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.synapse_plot import visualize, visualize_gif
from harmonium import Harmonium

"""
################################################################################
Harmonium (Restricted Boltzmann Machine) Exhibit File:

Samples a pre-trained harmonium (one that has been fit to a database, such as 
MNIST). Specifically, this utilizes a block Gibbs sampler to obtain 
confabulations/fantasies from the harmonium model.

Usage:
$ python sample_harmonium.py --dataX="/path/to/train_patterns.npy" \
                             --output_dir="/path/to/samples_directory/" 

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

################################################################################
## read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '',["dataX=", "output_dir=", "num_chains=", "init_mc_chains_from_data=", "seed=", 
                      "sample_persist_chains="]
)

seed = 117 ## seed value to control the seeding of JAX noise-creation process
init_mc_chains_from_data = False ## should Markov chains be initialized from randomly sampled data?
sample_persist_chains = True ## use persistent chains?
dataX = "../../data/mnist/trainX.npy" ## what dataset should be used? (should be same/similar to what RBM was fit to)
output_dir = "exp/samples/" ## output directory of this sampling script
num_chains = 3 ## number of Markov chains (default is 3 for demo)
for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--output_dir"):
        output_dir = arg.strip()
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--num_chains"):
        num_chains = int(arg.strip())
    elif opt in ("--init_mc_chains_from_data"):
        init_mc_chains_from_data = (arg.strip().lower() == "true")
    elif opt in ("--sample_persist_chains"):
        sample_persist_chains = (arg.strip().lower() == "true")
print("Data: ",dataX)
makedir(output_dir) ## create samples sub-dir if it doesn't already exist

def post_process_samples(raw_markov_chains, px=28, py=28, pixel_scale=255.): ## post-processes a batch of Markov chains
    batch_size = raw_markov_chains[0].shape[0]  ## get number of raw chains
    proc_samples = []  ## post-processed chains (2D tensors)
    proc_flat_samples = []  ## post-processed chains (flattened vecs)
    for i in range(batch_size):  ## for each chain
        proc_samples.append([])
        proc_flat_samples.append([])

    chain_len = len(raw_markov_chains) ## all chains are of same temporal length
    for s in range(chain_len): ## for each step
        for i in range(batch_size): ## for each chain
            x_s = raw_markov_chains[s][i:i+1, :] ## get s-th step, i-th sample vector
            proc_flat_samples[i].append(x_s)
            x_s = jnp.reshape(x_s, (px, py)) * pixel_scale ## map back to un-normalized pixel-space
            proc_samples[i].append(x_s)
    return proc_samples, proc_flat_samples

dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 4)

################################################################################
## load in seeding dataset and RBM model
X = jnp.load(dataX)
X = X * (X >= 0.45) ## binarize dataset
px = py = int(jnp.sqrt(X.shape[1])) #28

x_seed = None
if init_mc_chains_from_data:
    ## grab a seed batch of data to data-dependently initialize RBM Gibbs sampler
    ptrs = random.permutation(subkeys[2], X.shape[0])
    x_seed = X[ptrs[0:num_chains], :] ## seed via data

model = Harmonium(subkeys[0], load_dir="exp/")
print("--- RBM Synaptic Stats ---")
print(model.get_synapse_stats())

################################################################################
## set up MCMC sampling from the above RBM, and simulate chains
thinning_point = 20 #50 ## degree to which MCMC chains will be thinned (0 disables this)
n_samps = 9 * 9 #20 * 20 #144 #900 ## total number of total samples to collect from each chain
burn_in = 10 * thinning_point #500 * thinning_point ## MCMC burn-in (how many samples to discard per chain)
n_samp_steps = n_samps * thinning_point ## calculate number of total Markov chain steps to take
sample_buffer_maxlen = n_samps #-1

print("---Simulating Block Gibbs Sampling Chains ---")
if sample_persist_chains:
    print(" > Using persistent chains")
    x_mcmc_samps = model.sample_from_gibbs_chains(
        subkeys[1], n_steps=n_samp_steps, thinning_point=thinning_point, n_samples=num_chains, 
        sample_buffer_maxlen=sample_buffer_maxlen, verbose=True
    )
else:
    x_mcmc_samps = model.sample( ## run MCMC sampling chains to draw samples from RBM
        subkeys[1], n_steps=n_samp_steps, x_seed=x_seed, thinning_point=thinning_point, burn_in=burn_in,
        sample_buffer_maxlen=sample_buffer_maxlen, n_samples=num_chains, verbose=True
    )

################################################################################
## post-process chains (into videos and sample image grids)
x_chain_list, xflat_chain_list = post_process_samples(x_mcmc_samps, px=px, py=py) ## create list(s) of Markov chains
for i in range(num_chains):
    print(f"\r>>> Rendering Markov chain ({i})...", end="")
    x_raw = x_chain_list[i] ## grab one Markov chain
    xflat_raw = xflat_chain_list[i]
    xflat_raw = jnp.concat(xflat_raw, axis=0)

    visualize([xflat_raw.T], [(28, 28)], output_dir + f"samples_{i}")  ## create sample img grid
    visualize_gif(x_raw, path=output_dir, name=f'markov_chain_{i}') ## create gif
print()
