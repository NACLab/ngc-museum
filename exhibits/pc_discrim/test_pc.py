from ngcsimlib.controller import Controller
from jax import numpy as jnp, random, nn, jit
import sys

## bring in ngc-learn analysis tools
from ngclearn.utils.io_utils import makedir
#from ngclearn.utils.viz_utils import visualize, create_raster_plot
#from ngclearn.utils.patch_utils import generate_patch_set

@jit
def fx(x):
    #return nn.relu(x)
    return nn.tanh(x)

@jit
def dfx(x): ## this is needed to avoid bad initial conditions
    #return (x >= 0.).astype(jnp.float32) #return (x >= 0.).astype(jnp.float32)
    tanh_x = nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

def calc(x, W1, W2, K, beta): # analytical/hand-coded 2-layer NGC
    z2 = jnp.zeros([x.shape[0],W2.shape[0]])
    z1 = jnp.zeros([x.shape[0],W1.shape[0]])
    z0 = x
    e1 = jnp.zeros([x.shape[0],W1.shape[0]])
    e0 = jnp.zeros([x.shape[0],x.shape[1]])
    for k in range(K):
        # correct
        d2 = (jnp.matmul(e1,W2.T) * dfx(z2))
        z2 = z2 + d2 * beta
        z2_f = fx(z2)
        d1 = (jnp.matmul(e0,W1.T) * dfx(z1)) - e1
        #print(beta)
        #print(z1)
        z1 = z1 + d1 * beta
        # predict and check
        z1_f = fx(z1)
        z1_mu = jnp.matmul(z2_f, W2)
        e1 = z1 - z1_mu
        z0_mu = jnp.matmul(z1_f, W1)
        e0 = z0 - z0_mu
    dW2 = jnp.matmul(z2_f.T, e1)
    dW2 = -dW2 # flip direction to descent
    dW1 = jnp.matmul(z1_f.T, e0)
    dW1 = -dW1 # flip direction to descent
    return z0_mu, z1_mu, z1_f, z2_f, e0, e1, dW1, dW2

directory = "exp"

makedir(directory)
makedir(directory + "/filters")
makedir(directory + "/raster")
#makedir(directory + "/model")

_X = jnp.load("data/baby_mnist/babyX.npy")
_Y = jnp.load("data/baby_mnist/babyY.npy")
n_batches = _X.shape[0]

viz_mod = 1
n_samp_mod = 100 #50
mb_size = 1
n_iter = 10 #1
patch_shape = (3, 3)
in_dim = patch_shape[0] * patch_shape[1]
hid1_dim = 6 #100
hid2_dim = 4

T = 20 #200 #250 # num discrete time steps to simulate
dt = 1.
tau_m = 2. # ms

dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 10)

################################################################################
## Create model
model = Controller()
### set up neuronal cells
z0 = model.add_component("rate", name="z0", n_units=in_dim, tau_m=0.,
                         leakRate=0., key=subkeys[0])
e0 = model.add_component("error", name="e0", n_units=in_dim)
z1 = model.add_component("rate", name="z1", n_units=hid1_dim, tau_m=tau_m,
                         act_fx="tanh", leakRate=0., key=subkeys[1])
e1 = model.add_component("error", name="e1", n_units=hid1_dim)
z2 = model.add_component("rate", name="z2", n_units=hid2_dim, tau_m=tau_m,
                         act_fx="tanh", leakRate=0., key=subkeys[2])
### set up connecting synapses
W1 = model.add_component("hebbian", name="W1", shape=(hid1_dim, in_dim),
                         eta=0.01, wInit=("uniform", -0.3, 0.3), w_bound=0.,
                         signVal=-1., key=subkeys[3])
W2 = model.add_component("hebbian", name="W2", shape=(hid2_dim, hid1_dim),
                         eta=0.01, wInit=("uniform", -0.3, 0.3), w_bound=0.,
                         signVal=-1., key=subkeys[4])
E1 = model.add_component("hebbian", name="E1", shape=(in_dim, hid1_dim),
                         eta=0., wInit=("uniform", -0.3, 0.3), w_bound=0.,
                         signVal=-1., key=subkeys[3])
E2 = model.add_component("hebbian", name="E2", shape=(hid1_dim, hid2_dim),
                         eta=0., wInit=("uniform", -0.3, 0.3), w_bound=0.,
                         signVal=-1., key=subkeys[4])
## wire z1 to e0 via W1
model.connect(z1.name, z1.outputCompartmentName(), W1.name, W1.inputCompartmentName())
model.connect(W1.name, W1.outputCompartmentName(), e0.name, e0.meanName())#, bundle="additive")
model.connect(z0.name, z0.rateActivityName(), e0.name, e0.targetName())
## wire z2 to e1 via W2
model.connect(z2.name, z2.outputCompartmentName(), W2.name, W2.inputCompartmentName())
model.connect(W2.name, W2.outputCompartmentName(), e1.name, e1.meanName())
model.connect(z1.name, z1.rateActivityName(), e1.name, e1.targetName())
## wire e0 to z0 via d/dz0
#model.connect(e0.name, e0.derivTargetName(), z0.name, z0.inputCompartmentName())#, bundle="additive")
## wire e0 to z1 via W1.T and e1 to z1 via d/dz1
model.connect(e0.name, e0.derivMeanName(), E1.name, E1.inputCompartmentName())
model.connect(E1.name, E1.outputCompartmentName(), z1.name, z1.inputCompartmentName())#, bundle="additive")
model.connect(e1.name, e1.derivTargetName(), z1.name, z1.pressureName())#, bundle="additive")
## wire e1 to z2 via W2.T
model.connect(e1.name, e1.derivMeanName(), E2.name, E2.inputCompartmentName())
model.connect(E2.name, E2.outputCompartmentName(), z2.name, z2.inputCompartmentName())#, bundle="additive")

## setup W1 for its 2-factor Hebbian update
model.connect(z1.name, z1.outputCompartmentName(), W1.name, W1.presynapticCompartmentName())
model.connect(e0.name, e0.derivMeanName(), W1.name, W1.postsynapticCompartmentName())
## setup W2 for its 2-factor Hebbian update
model.connect(z2.name, z2.outputCompartmentName(), W2.name, W2.presynapticCompartmentName())
model.connect(e1.name, e1.derivMeanName(), W2.name, W2.postsynapticCompartmentName())

## checks that everything is valid within model structure
#model.verify_cycle()

## make key commands known to model
## will need to clamp to z0 and e0.target = x; get e0 to do x - 0 at t = 0?
model.add_command("reset", command_name="reset",
                  component_names=[z0.name, z1.name, z2.name,
                              e0.name, e1.name],
                  reset_name="do_reset")
model.add_command(
    "advance", command_name="advance",
    component_names=[E1.name, E2.name,
                     z0.name, z1.name, z2.name,
                     W1.name, W2.name,
                     e0.name, e1.name
                    ]
)
model.add_command("evolve", command_name="evolve", component_names=[W1.name, W2.name])
model.add_command("clamp", command_name="clamp_input",
                  component_names=[z0.name], compartment=z0.inputCompartmentName(),
                  clamp_name="x")
model.add_command("clamp", command_name="clamp_target",
                  component_names=[e0.name], compartment=e0.targetName(),
                  clamp_name="target")
model.add_command("save", command_name="save", component_names=[W1.name, W2.name],
                  directory_flag="dir")

## tell model the order in which to run automatic commands
# myController.add_step("clamp_input")
model.add_step("advance")
#model.add_step("evolve")
################################################################################

Xb = random.uniform(subkeys[8], (1, in_dim), dtype=jnp.float32)

## run analytical model
W1 = model.components["W1"].weights
W2 = model.components["W2"].weights
z0_mu, z1_mu, z1_f, z2_f, e0, e1, dW1, dW2 = calc(Xb, W1, W2, K=T, beta=dt/tau_m)
# print("+++++++++++++++++ Ref ++++++++++++++++++++++")
# print("z2: ",z2_f)
# print("mu1: ",z1_mu)
# print("e1: ",e1)
# print("z1: ",z1_f)
# print("mu0: ",z0_mu)
# print("e0: ",e0)
# print("++++++++++++++++++++++++++++++++++++++++++++")

model.reset(do_reset=True)
model.components["E1"].weights = (model.components["W1"].weights).T
model.components["E2"].weights = (model.components["W2"].weights).T
for ts in range(0, T):
    #print("###################### {} #########################".format(ts))
    model.clamp_input(x=Xb) #x=inp) ## clamp data to z0
    #model.clamp_target(target=Xb) ## clamp data to e0.target
    #print("e0.comp = ",model.components["e0"].compartments)
    model.runCycle(t=ts*dt, dt=dt)
    #print("e0.comp = ",model.components["e0"].compartments)
    #sys.exit(0)

    # print("z2: ",model.components["z2"].rateActivity)
    # print("mu1: ",model.components["e1"].mean)
    # print("e1: ",model.components["e1"].derivMean)
    # print("z1: ",model.components["z1"].rateActivity)
    # print("mu0: ",model.components["e0"].mean)
    # print("e0: ",model.components["e0"].derivMean)

    #print("e0.comp = ",model.components["e0"].compartments)
    #print(model.components["W1"].outputCompartment)

# print("z2: ",model.components["z2"].activity)
# print("mu1: ",model.components["e1"].mean)
# print("e1: ",model.components["e1"].derivMean)
# print("z1: ",model.components["z1"].activity)
# print("mu0: ",model.components["e0"].mean)
# print("e0: ",model.components["e0"].derivMean)

print("z2: ",jnp.sum(jnp.abs(model.components["z2"].activity - z2_f)))
print("mu1: ",jnp.sum(jnp.abs(model.components["e1"].mean - z1_mu)))
print("e1: ",jnp.sum(jnp.abs(model.components["e1"].derivMean - e1)))
print("z1: ",jnp.sum(jnp.abs(model.components["z1"].activity - z1_f)))
print("mu0: ",jnp.sum(jnp.abs(model.components["e0"].mean - z0_mu)))
print("e0: ",jnp.sum(jnp.abs(model.components["e0"].derivMean - e0)))
#sys.exit(0)
model.evolve(t=T, dt=dt)

print("dW1: ",jnp.sum(jnp.abs(model.components["W1"].Eg - dW1)))
print("dW2: ",jnp.sum(jnp.abs(model.components["W2"].Eg - dW2)))
