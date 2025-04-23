from ngclearn.utils.io_utils import makedir
import sys, getopt as gopt, optparse, time
from ngclearn.utils.viz.synapse_plot import visualize, visualize_gif
from jax import numpy as jnp, random, jit, nn
from snn import SNN
from rat_maze import RatMaze

"""
################################################################################
T-Maze Spiking Neural Network Controller Exhibit File:

Adapts a simple SNN controller to navigate the rat T-Maze problem. The goal is 
to find the "food" in the top-right corner of the environment. Synaptic 
efficacies are evolved via modulated spike-timing-dependent plasticity (using 
eligibility traces).

Usage:
$ python sim_ratmaze.py --seed=1234 --is_random=False --results_dir=/output_dir/ 

@author: The Neural Adaptive Computing Laboratory
################################################################################
"""

def print_syn_stats(agent): ## print synaptic statistics
    W2, W1 = agent.circuit.get_components("W2", "W1")
    _W2 = W2.weights.value
    _W1 = W1.weights.value
    print(
        f"W1 | mu: {jnp.mean(_W1):.5f} sd: {jnp.std(_W1):.5f} min: {jnp.amin(_W1):.5f} max {jnp.amax(_W1):.5f} n: {jnp.linalg.norm(_W1):.5f}"
    )
    print(
        f"W2 | mu: {jnp.mean(_W2):.5f} sd: {jnp.std(_W2):.5f} min: {jnp.amin(_W2):.5f} max {jnp.amax(_W2):.5f} n: {jnp.linalg.norm(_W2):.5f}"
    )
    print("++++++++++++++++")

results_dir = ""
is_random = True
is_verbose = False
seed = 1234
# read in general program arguments
options, remainder = gopt.getopt(sys.argv[1:], '', ["seed=", "is_verbose=", "is_random=", "results_dir="])

for opt, arg in options:
    if opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--is_verbose"):
        is_verbose = (arg.strip().lower() == "true")
    elif opt in ("--is_random"):
        is_random = (arg.strip().lower() == "true")
    elif opt in ("--results_dir"):
        results_dir = arg.strip()
makedir(results_dir)
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 4)

## create maze space
n_a = 4 ## number actions -- four cardinal directions
n_s = 2 ## number of physical state sensors
n_d = 3 ## size of maze / number of encoded dimensions
n_eval_episodes = 1 #0
n_steps = 30
n_episodes = 200

render = False
maze = RatMaze(
    dkey=subkeys[0], width=n_d, reward_type="dist", is_deterministic=True, episode_len=n_steps, maze_type="t_maze"
)
maze.reset()

field_shape = (maze.dim_x, maze.dim_y)
N_i = maze.get_dim()
N_h = 36 ## number of projection/hidden neurons
N_o = n_a ## number of outputs/action units

viz_rfields = False
warmup = False
## epsilon-greedy policy meta-parameters
p_decay = 0.975
p_rand = p_min = 0.02 ## hold eps prob constant
norm_viz_mod = 50
agent = SNN(dkey=subkeys[1], in_dim=N_i, out_dim=N_o, n_hid=N_h)

if viz_rfields:
    print_syn_stats(agent)

tag = "snn"
if is_random:
    tag = "rand"

n_episodes = n_episodes + n_eval_episodes
eval_model = False
reward_history = []
completion_history = []
length_history = [] ## episode lengths
n_completions = 0
epi_frames = []
for e in range(n_episodes):
    obs_t = maze.reset()

    if e > n_episodes - (n_eval_episodes+1):
        eval_model = True

    rendered_steps = []
    frames = []

    agent.set_to_resting_state() ## snap SNN to resting state values
    if warmup:
        enc_t = maze.encode(jnp.expand_dims(obs_t, axis=0))
        act_spikes = agent.infer(enc_t)  ## process sensory input
    if render or eval_model:
        #frame = maze.render()
        #print(frame)
        frames.append(maze.render(raw_env_pixels=True))

    r_epi = 0. ## episodic reward
    L_epi = 0 ## episodic length
    act_dist = 0.
    num_blanks = 0
    complete = 0.

    for i in range(n_steps):
        dkey, *subkeys = random.split(dkey, 4)
        action = 0
        if is_random: ## random agent
            action = random.randint(subkeys[0], (1,), minval=0, maxval=n_a)[0]
        else: ## snn agent
            agent.set_to_resting_state() ## (snap to resting potentials)
            enc_t = maze.encode(jnp.expand_dims(obs_t, axis=0))
            stats = None
            if eval_model:
                act_spikes = agent.infer(enc_t)  ## infer over sensory input
            else:
                act_spikes = agent.process(enc_t)  ## process sensory input
            freq = jnp.sum(act_spikes, axis=0)
            W1, W2 = agent.circuit.get_components("W1", "W2")

            if jnp.sum(freq) <= 0.:
                num_blanks += 1
            action = jnp.argmax(freq)  ## get action
            if random.uniform(subkeys[0], (1,))[0] <= p_rand:
                action = random.randint(subkeys[1], (1,), minval=0, maxval=n_a)[0]

        act_dist = nn.one_hot(action, num_classes=n_a) + act_dist
        obs_t, reward, done, _ = maze.step(action)
        L_epi += 1
        if not done:
            r_epi -= 1.

        if not is_random:
            if not eval_model:
                r_t = reward
                agent.adapt(r_t)  ## modulate plasticity via r-stdp-et
        if render or eval_model: ## carry out any rendering
            #frame = maze.render()
            #print(frame)
            frames.append(maze.render(raw_env_pixels=True))
        if done: ## terminate episode if done signal received
            complete = 1.
            n_completions += 1
            break

    reward_history.append(r_epi)
    completion_history.append(complete)
    length_history.append(L_epi)
    r_mu = jnp.mean(jnp.array(reward_history)[-100:])
    L_mu = jnp.mean(jnp.array(completion_history)[-100:])
    if is_verbose:
        print(
            f"{e}: r.mu = {r_mu:.4f} acc.mu = {L_mu:.4f}  act: {act_dist} = {jnp.sum(act_dist)}  G = {maze._goal_pos}  nB {num_blanks} "
        )
    else:
        print(
            f"{e}: r.mu = {r_mu:.4f} acc.mu = {L_mu:.4f} "
        )

    p_rand = jnp.maximum(p_rand * p_decay, p_min)

    if viz_rfields and e % norm_viz_mod == 0:
        print_syn_stats(agent)
        _W1, _W2 = agent.circuit.get_components("W1", "W2")
        _W1 = _W1.weights.value
        _W2 = _W2.weights.value
        visualize([_W1, _W2], [field_shape, (int(jnp.sqrt(N_h)), int(jnp.sqrt(N_h)))], "rec_fields.png")

    if render or eval_model:
        print(f"Saving vid for episode {e} to disk...")
        visualize_gif(frames, path='.', name=f'episode{e}')

if viz_rfields:
    print_syn_stats(agent)
    _W1, _W2 = agent.circuit.get_components("W1", "W2")
    _W1 = _W1.weights.value
    _W2 = _W2.weights.value
    visualize([_W1, _W2], [field_shape, (int(jnp.sqrt(N_h)), int(jnp.sqrt(N_h)))], "rec_fields.png")

jnp.save(f"{results_dir}{tag}_lengths_{seed}.npy", jnp.array(length_history))
jnp.save(f"{results_dir}{tag}_returns_{seed}.npy", jnp.array(reward_history))
jnp.save(f"{results_dir}{tag}_completes_{seed}.npy", jnp.array(completion_history))

