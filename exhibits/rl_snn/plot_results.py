import matplotlib #.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cmap = plt.cm.jet
import sys, getopt as gopt, optparse, time
import numpy as np

"""
################################################################################
Plots reward/completion trajectories produced by `sim_ratmaze.py`.

Usage:
$ python plot_results.py --result_type=item_type --results_dir=results_directory \
                         --seeds="space_separated_seedstring_list"

Note: `item_type` can only either be 'returns' or 'completes'

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

def smoothen(data, window_len=100): ## perform Rainbow RL-style smoothening over reward trajectory
    outs = []
    win = []
    for i in range(data.shape[0]):
        xi = data[i]
        win.append(xi)
        if len(win) > window_len:
            win.pop(0)
        mu = np.mean(np.array(win))
        outs.append(mu)
    return np.array(outs)

#result_type = "returns" #
result_type = "completes"
results_dir = "results/ratmaze/"
seeds = ["59", "1234", "44", "816"]

# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["result_type=", "results_dir=", "seeds="])

for opt, arg in options:
    if opt in ("--result_type"):
        result_type = arg.strip().replace(" ","")
        if result_type != "returns" and result_type != "completes":
            print("ERROR: Only `returns` or `completes` plotted!")
            sys.exit(1)
    elif opt in ("--results_dir"):
        results_dir = arg.strip()
    elif opt in ("--seeds"):
        seeds = arg.strip().lower().split(" ")

fig, ax = plt.subplots()
output_fname = f"{result_type}.jpg"

snn = []
rand = []
for seed in seeds:
    returns = np.squeeze(np.load(f"{results_dir}snn_{result_type}_{seed}.npy"))
    max_len = returns.shape[0]
    snn.append(np.expand_dims(smoothen(returns), axis=1))

    rand_returns = np.squeeze(np.load(f"{results_dir}rand_{result_type}_{seed}.npy"))[0:max_len]
    rand.append(np.expand_dims(smoothen(rand_returns), axis=1))

snn = np.concatenate(snn, axis=1)
rand = np.concatenate(rand, axis=1)
snn_mu = np.mean(snn, axis=1)
#print(snn_mu)
snn_sd = np.std(snn, axis=1)
rand_mu = np.mean(rand, axis=1)
rand_sd = np.std(rand, axis=1)

epi_values = np.arange(start=0, stop=snn_mu.shape[0])

a = ax.plot(epi_values, snn_mu, linestyle='-.', color='red', linewidth=3) #markersize=10  marker='s'
lb = np.squeeze(snn_mu - snn_sd)
ub = np.squeeze(snn_mu + snn_sd)
if result_type == "completes":
    lb = np.maximum(0., lb)
    ub = np.minimum(1., ub)
ax.fill_between(epi_values, lb, ub, color="red", alpha=0.2)
b = ax.plot(epi_values, rand_mu, linestyle='--', color='blue', linewidth=3) #, alpha=.5)
ax.fill_between(epi_values, np.squeeze(rand_mu - rand_sd), np.squeeze(rand_mu + rand_sd), color="blue", alpha=0.2)

plot_type = "Returns over Time"
yaxis_lab = "Episodic Reward"
if result_type == "completes":
    plot_type = "Task Accuracy"
    yaxis_lab = "Completions"
ax.set(
    xlabel='Episode', ylabel=yaxis_lab,
    title=plot_type
)
#ax.legend([a[0]],['Returns'])
ax.legend([a[0],b[0]],['SNN R','Rand R'])
ax.grid()
fig.savefig(output_fname)
plt.close()

