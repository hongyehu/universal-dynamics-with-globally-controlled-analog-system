# %%

import pennylane as qml
import os
import pennylane.numpy as np
from jax.experimental.ode import odeint
import jax.numpy as jnp
import jax
from jax import random
import optax
import sys

import GRAPE_HelperFunctions as hf

# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
path = os.getcwd()
folder_param = path + '/Optimal Parameters/'
os.makedirs(folder_param, exist_ok=True)
folder_loss = path + '/Loss/'
os.makedirs(folder_loss, exist_ok=True)
folder_haar = path + '/Haar Random States/'
os.makedirs(folder_haar, exist_ok=True)

########################################################################
########################################################################
# In this version of the code, we use a nn + nnn model for the Rydberg 
# Hamiltonian during training. The atoms are placed at equally spaced 
# points along a 1D chain (open BC). The target Hamiltonian is encoded 
# in int_tags and int_weights. The target Hamiltonian is assumed to be 
# nearest neighbor. 


N = 3   # system size

C6 = 862690 * 2 * jnp.pi    
dev = qml.device("default.qubit", wires=range(N))

####### Parameters for simulation times: 
tau = 1.25   # (physical) simulation time
dtau = 0.05  # time resolution for pulse
iX = 8.9     # initial nn placement (um)

t_vec = jnp.array([0.8])
tau_vec = jnp.arange(0.0, tau + dtau, dtau)

RandSpread = 1.0    # spread used for random parameter initialization 
                    # Omega: (0, spread) 
                    # Delta: (-spread, spread)

####### Parameters for target Hamiltonian: 
int_tags = ['ZXZ']
int_weights = [1.0]


####### Parameters for pulse parameterization: 
n_T = len(tau_vec) - 2          # number of parameters (beginning and end of pulse fixed to be zero)

params = jnp.zeros(2*n_T)       # total number of parameters (includes nn spacing)
iX = 8.9                        # atomic spacing

init_str = "_fixedX" + str(iX) + '_Rand' + str(RandSpread)
key = random.PRNGKey(np.random.choice(100000))
params = jax.lax.concatenate([RandSpread*random.uniform(key, shape=(n_T,)), 2*RandSpread*random.uniform(key, shape=(n_T,)) - RandSpread], dimension=0)

####### Parameters for optimization: 
lr = 1e-2           # learning rate
R_ddu = 1e-4        # regularization on second derivative of control (to encourage smoothness)

num_states = 50     # this sets the number of Haar random states that are used during training
n_iter = 20000      # maximum number of iterations
thresh = 0.00001    # threshold to stop optimization (percent loss change every 100 iterations)
eps = 1e-6          # threshold to stop optimization (loss -- that is, average infidelity + penalties)

ham_tag = ''
ham_label = 'N = ' + str(N) + ','
for x_ind, x in enumerate(int_tags):
    ham_tag = ham_tag + '_' + str(int_weights[x_ind]) + x
    if str(int_weights[x_ind])[0] == '-' or x_ind == 0:
        ham_label = ham_label  + str(int_weights[x_ind]) + ' ' + x
    else:
        ham_label = ham_label + '+' + str(int_weights[x_ind]) + ' ' + x
    if len(x) > N:
        sys.exit("Not enough qubits to simulate this interaction!")
ham_label = '$' + ham_label + '$'
state_str = "-HR" + str(num_states)

@qml.qnode(dev, interface="jax")
def qr_haar_random_unitary(N):
    qml.QubitUnitary(hf.qr_haar(2**N), wires=range(N))
    return qml.state()

def fetch_initial_state(N, tag=0):
    filename = 'N' + str(N) + '-' + str(tag) + '.npy'
    filepath = folder_haar + filename
    if os.path.isfile(filepath):
        state = np.load(filepath)
    else:
        state = qr_haar_random_unitary(N)
        np.save(filepath, np.array(state))
    return state

def PiecewiseConstant(p, t):
    t_points = jnp.array([dtau * k for k in range(len(p))])
    return jnp.interp(t, t_points, p)

def target_ham(tp, N=N, int_tags=int_tags, int_weights=int_weights):
    # This function returns the target Hamiltonian 
    op_zeros = [0 for x in range(N)]
    n_ints = len(int_tags)
    ints = [jnp.zeros((2 ** N, 2 ** N)).astype(jnp.complex128) for x in range(n_ints)]

    for int_ind, int_tag in enumerate(int_tags):
        for i in range(N - len(int_tag) + 1):
            op_list = op_zeros[:]
            for P_ind, P in enumerate(int_tag):
                op_list[i + P_ind] = hf.str_to_pauli(int_tag[P_ind])
            ints[int_ind] += hf.get_op_mat(op_list)

    H = jnp.zeros((2 ** N, 2 ** N)).astype(jnp.complex128)
    for int_ind, interaction in enumerate(ints):
        H += int_weights[int_ind] * interaction

    return H


def native_ham_jax(tau_p, params, iX=iX):
    # This Rydberg hamiltonian only includes nn and nnn interactions
    O_p = jax.lax.concatenate([jnp.zeros(1), params[:n_T], jnp.zeros(1)], dimension=0)
    D_p = jax.lax.concatenate([jnp.zeros(1), params[n_T:2*n_T], jnp.zeros(1)], dimension=0)
    
    Omega = PiecewiseConstant(O_p, tau_p)
    Delta = PiecewiseConstant(D_p, tau_p)
    
    dX = iX
    op_zeros = [0 for x in range(N)]
    total_I = hf.get_op_mat(op_zeros)
    X_tot = jnp.zeros((2 ** N, 2 ** N)).astype(jnp.complex128)
    n_tot = jnp.zeros((2 ** N, 2 ** N)).astype(jnp.complex128)
    nn_tot = jnp.zeros((2 ** N, 2 ** N)).astype(jnp.complex128)
    for i in range(N):
        op_list = op_zeros[:]
        op_list[i] = 1
        X_tot += hf.get_op_mat(op_list)
        op_list[i] = 3
        n_tot += total_I - hf.get_op_mat(op_list)
        if i == 1: 
            j_vec = [i - 1]
        elif i > 1:
            j_vec = [i - 2, i - 1] # only nn and nnn
        else: 
            j_vec = []
        for j in j_vec:
            op_list_i = op_zeros[:]
            op_list_i[i] = 3
            op_list_j = op_zeros[:]
            op_list_j[j] = 3
            op_list_ij = op_zeros[:]
            op_list_ij[i] = 3
            op_list_ij[j] = 3
            d_ij = dX * (i - j)
            coeff = (C6 / 4) * ((d_ij) ** (-6))
            nn_tot += coeff * (total_I - hf.get_op_mat(op_list_i) - hf.get_op_mat(op_list_j) + hf.get_op_mat(op_list_ij))

    H = ((Omega / 2) + 0.0j) * X_tot + (-(Delta / 2) + 0.0j) * n_tot + nn_tot
    return H


def evolve_batch(states, h_fun, tau_p: jnp.array, *args, **solver_kws):
    # time evolve a batch of states under a (time-dependent) hamiltonian h_fun
    def f(y, tau_p, *args):
        h = -1.0j * h_fun(tau_p, *args)
        return h @ y  # derivative = -i H*state
    ts = jnp.concatenate((jnp.array([0.0]), tau_p))
    s1 = odeint(f, states, ts, *args, **solver_kws)
    return s1


initial_states_batch = jnp.zeros((2**N, num_states), dtype=jnp.complex128)
for ic_tag in range(num_states):
    initial_sim_state = fetch_initial_state(N, tag=ic_tag)
    initial_states_batch= initial_states_batch.at[:, ic_tag].set(initial_sim_state)

def evolve_batch_ex_wrapper(t):
    s1 = evolve_batch(initial_states_batch, target_ham, jnp.array(t))
    return s1[1:, :, :]

ex_states_batch = evolve_batch_ex_wrapper(t_vec)

# indices for ex_states_batch: time, state components, initial state index

def penalty_term(x, x_min=0.0, x_max=13.5):
    below_min = jnp.where(x < x_min, (x_min - x) ** 2, 0.0)
    above_max = jnp.where(x > x_max, (x - x_max) ** 2, 0.0)
    penalty = below_min + above_max
    return jnp.sum(penalty)

iter = 0
@jax.jit
def loss(params, ex_t_states, lambda_max=100): # average fidelity of random initial states
    s1 = evolve_batch(initial_states_batch, native_ham_jax, jnp.array([tau]), params)
    final_states = s1[-1, :, :]
    overlap = jnp.diag(jnp.conj(ex_t_states.T) @ final_states)
    fids = jnp.mean(jnp.real(overlap * jnp.conj(overlap)))
    penalty1 = penalty_term(params[:n_T], x_min=0.0, x_max=15.7)
    penalty2 = penalty_term(params[n_T:2*n_T], x_min=-100.0, x_max=100.0)
    O_p = jax.lax.concatenate([jnp.zeros(1), params[:n_T], jnp.zeros(1)], dimension=0)
    D_p = jax.lax.concatenate([jnp.zeros(1), params[n_T:2*n_T], jnp.zeros(1)], dimension=0)
    dO = jnp.gradient(O_p, dtau)
    ddO = jnp.gradient(dO, dtau)
    dD = jnp.gradient(D_p, dtau)
    ddD = jnp.gradient(dD, dtau)
    penalty3 = penalty_term(dO, x_min=-270.0, x_max=270.0)
    penalty4 = penalty_term(dD, x_min=-2*270.0, x_max=2*270.0)
    penalty5 = penalty_term(ddO, x_min=-10*270.0, x_max=10*270.0)
    penalty6 = penalty_term(ddD, x_min=-10*2*270.0, x_max=10*2*270.0)
    return 1.0 - fids + lambda_max * (penalty1 + penalty2 + penalty3 + penalty4 + penalty5 + penalty6) + R_ddu * (jnp.mean(ddO ** 2) + jnp.mean(ddD ** 2))


@jax.jit
def fids(params, ex_t_states):
    s1 = evolve_batch(initial_states_batch, native_ham_jax, jnp.array([tau]), params)
    final_states = s1[-1, :, :]
    overlap = jnp.diag(jnp.conj(ex_t_states.T) @ final_states)
    fids = jnp.mean(jnp.real(overlap * jnp.conj(overlap)))
    return fids


value_and_grad = jax.jit(jax.value_and_grad(loss))
loss_t = []
fid_vec = []
param_hist = []
optimizer = optax.adam(learning_rate=lr)

for t_ind, t in enumerate(t_vec):
    t = jnp.round(t, 2)
    print(f"t: {jnp.round(t, 2)}")
    filename = "N" + str(N) + ham_tag + "_r" + str(R_ddu)  + "_lr" + str(lr) + "_nT" + str(n_T) \
               + init_str + "_tau" + str(tau) + "_t" + \
               str(t) + state_str + ".npy"

    filepath_param = folder_param + filename
    filepath_loss = folder_loss + filename

    ex_t_states = ex_states_batch[t_ind, :, :]
    if not os.path.isfile(filepath_param):
        print(filename)
        
        ## Optimization loop
        params = jax.lax.concatenate([RandSpread*random.uniform(key, shape=(n_T,)), 2*RandSpread*random.uniform(key, shape=(n_T,)) - RandSpread], dimension=0)
        opt_state = optimizer.init(params)
        loss_vec = []
        gradients = []
        change = 1.0
        last_loss = 10.0
        loss_val = 10.0
        best_loss = 10.0
        best_params = params.copy()
        iter = 0
        while change > thresh and loss_val > eps and iter <= n_iter:
            loss_val, grad_circuit = value_and_grad(params, ex_t_states)
            updates, opt_state = optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            gradients.append(np.mean(np.abs(grad_circuit)))
            loss_vec.append(loss_val)
            if loss_val < best_loss:
                best_params = params.copy()
                best_loss = loss_val.copy()
            if not iter % 100:
                change = np.abs((last_loss - loss_val) / loss_val)
                last_loss = loss_val
                print(f"{iter}; loss: {loss_val}")
                print(f"mean grad: {gradients[-1]}")

            iter += 1
        best_fid = fids(best_params, ex_t_states)
        param_hist.append(best_params.copy())
        loss_t.append(loss_vec[-1])
        fid_vec.append(best_fid)
        np.save(filepath_param, best_params.copy())
        np.save(filepath_loss, loss_vec)
        print(f"Final fid: {best_fid}")

    else:
        print("Already exists!")
        print(filepath_param)
        params = np.load(filepath_param)
        loss_vec = np.load(filepath_loss)
        val, grad_circuit = value_and_grad(params, ex_t_states)
        param_hist.append(params.copy())
        loss_t.append(val)



# %%

import matplotlib.pyplot as plt
plt.rc("font", family = 'serif', size = 10.0)
plt.rc("figure", dpi = 600)
plt.rc("lines", linewidth = 1.2)
plt.rc('axes', axisbelow=True)
plt.rcParams.update({"text.usetex": True})
colors = [(0.25, 0.45, 0.6), (0.65, 0.3, 0.35), (0.35, 0.2, 0.35), (0.7, 0.55, 0.6), (0.75, 0.45, 0.55)]

fig, ax = plt.subplots(2, 1, sharex="all", figsize=(4, 3))

k = 0
t = np.round(t_vec[k], 2)
O_p = jax.lax.concatenate([jnp.zeros(1), param_hist[k][:n_T], jnp.zeros(1)], dimension=0)
D_p = jax.lax.concatenate([jnp.zeros(1), param_hist[k][n_T:2*n_T], jnp.zeros(1)], dimension=0)
ax[0].plot(tau_vec, O_p, label=f'$t = {"{:0.2f}".format(t, decimals=1)}$', color=colors[k])
ax[1].plot(tau_vec, D_p, color=colors[k])
ax[0].grid()
ax[0].legend()
ax[1].grid()
ax[1].set_xlabel(r'$\tau$')
ax[0].set_ylabel(r'$\Omega(\tau)$')
ax[1].set_ylabel(r'$\Delta(\tau)$')
ax[0].set_title(r'{}'.format(ham_label))
fig.tight_layout(pad=0.5)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(jnp.arange(len(loss_vec)), loss_vec)
ax.grid()
ax.set_xlabel(r'$N_{iter}$')
ax.set_ylabel(r'$\mathcal{L}$')
ax.set_title(r'{}'.format(ham_label))
fig.tight_layout(pad=0.5)
plt.show()
