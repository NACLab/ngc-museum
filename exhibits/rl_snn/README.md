# Spiking Neural Controller for Operant Conditioning

<b>Version</b>: ngclearn==2.0.0, ngcsimlib==1.0.0

This exhibit contains an implementation of a spiking neural network trained via modulated spike-timing-dependent plasticity (with eligibility traces) for simple reinforcement learning tasks (such as rat-maze navigation). This model embodies a few elements taken from: 

```
Chevtchenko, Sérgio F., and Teresa B. Ludermir. "Learning from sparse and delayed rewards with a multilayer spiking neural network." 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020.
```

<!--
<p align="center">
  <img height="150" src="fig/snn_arch.jpg"><br>
  <i>Visual depiction of the SNN controller architecture.</i>
</p>
-->

<!--
This model is also discussed in the ngc-learn
<a href="https://ngc-learn.readthedocs.io/en/latest/museum/rl_snn.html">documentation</a>.
-->

## Running the Model's Simulation

To train this implementation of an MSTDP-ET-driven SNN, simply run:

```console
$ python sim_ratmaze.py --seed=1234 --is_random=False \
                        --results_dir=/output_dir/ 
```

noting that if `is_random=True`, then a random control policy will be simulated (instead of the SNN model). Make sure you create the relevant results directory you provide as the argument to `results_dir`.

Alternatively, you may run the convenience bash script:

```console
$ ./sim.sh
```

which will execute and run the model simulation for the rat T-maze problem. Note that the bash script will run the simulation for four uniquely-seed trials and will further plot (and save to disk) the resulting four-trial reward and task accuracy curves for the simulated agent.

## Description

This model is effectively made up of three layers -- a sensory input layer (i.e., a Poisson process encoding layer), and two layers of (noisy) leaky integrate-and-fire neurons (one hidden layer and one output/control layer). The sensory layer connects to the hidden via fixed, random synaptic synapses (either -1, 0, or 1) whereas the hidden layer connects to the output/control layer via a modulated spike-timing-dependent plasticity (w/ eligibility traces), or MSTDP-ET, set of synapses. Note that both the hidden and output/control layers are recurrently wired via fixed inhibitory synapses.

<i>Task</i>: This model engages in reinforcement learning, learning
to navigate a simple rodent T-maze where the goal is for it (starting at the bottom of the T-maze) to find the "food" object (marked with an X as the goal state in our implemented environment).

## Hyperparameters

This model requires the following hyperparameters to obtain reasonable behavior across experimental trials:

```
T = 100 (number of time steps to simulate)
dt = 1 ms (integration time constant)
n_steps = 30 (number of episodic steps to simulate)
n_episodes = 200 (number of total episodes in a trial to carry out)
N_h = 36 (number of hidden neurons for the SNN model to use)
batch_size = 1
```

Note that there are other hyper-parameters embedded in the `snn.py` file (containing the `SNN` agent class), which govern the specific neuronal and plasticity dynamics.
