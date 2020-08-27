import matplotlib.pyplot as plt
import numpy as np

#from mpi4py import MPI

from params import DEFAULT_PARAMS
from settings import DEFAULT_MODEL, VALID_MODELS, NUM_RXN, UPDATE_DICTS, NUM_STEPS, INIT_CONDS, STATE_SIZE, RECEPTOR_STATE_SIZE
from formulae import ns_mean, ns_var

def update_state(state_timeseries, rxn_idx, step, model):
    """
    Update the state based on the increment associated with the rxn_idx of specified model
    """
    current_state = state_timeseries[step]
    increment = UPDATE_DICTS[model][rxn_idx]
    state_timeseries[step + 1, :] = current_state + increment
    return state_timeseries


def propensities(state, model, params=DEFAULT_PARAMS):
    """
    Computes propensitiers of each rxn based on current state.
    Reaction labelling follows order defined in UPDATE_DICTS.
    """
    p=params
    assert model in VALID_MODELS
    propensities = np.zeros(NUM_RXN[model])
    if model == 'mode_1':
        propensities[0] = p.k_on * p.c * (1 - state[0])  # bind (0.0 if already bound)
        propensities[1] = p.k_off * state[0]             # unbind (0.0 if already unbound)
        propensities[2] = p.k_p * state[0]               # produce one n molecule
        propensities[3] = p.d_n * state[1]               # degradation n molecule
    elif model == 'mode_2':
        propensities[0] = p.k_on * p.c * (1 - state[0])  # bind (0.0 if already bound) and produce one m molecule
        propensities[1] = p.k_off * state[0]             # unbind (0.0 if already unbound)
        propensities[2] = p.d_m * state[1]               # degradation m molecule
    elif model == 'combined':
        propensities[0] = p.k_on * p.c * (1 - state[0])  # bind (0.0 if already bound)
        propensities[1] = p.k_off * state[0]             # unbind (0.0 if already unbound)
        propensities[2] = p.k_p * state[0]               # produce one n molecule
        propensities[3] = p.d_n * state[1]               # degradation n molecule
        propensities[4] = p.d_m * state[2]
    elif model == 'kpr':
        propensities[0] = p.k_on * p.c * (1 - state[0]) * (1 - state[1])  # binding
        propensities[1] = p.k_off * state[0]                              # unbinding
        propensities[2] = p.k_f * state[0]                                # kpr forward step + GPCR event
        propensities[3] = p.k_off * state[1]                              # fall off
        propensities[4] = p.k_p * state[1]                                # produce n
        propensities[5] = p.d_n * state[2]                                # degradation n molecule
        propensities[6] = p.d_m * state[3]
    elif model == 'two_ligand_kpr':
        zero_indic = (1 -state[0]) * (1 -state[1]) * (1 -state[2]) * (1 -state[3])
        propensities[0] = p.k_on * p.c1 * zero_indic                      # binding of ligand #1
        propensities[1] = p.k_on * p.c2 * zero_indic                      # binding of ligand #2
        propensities[2] = p.k_off_1 * state[0]				              # unbinding of ligand #1 from state 1
        propensities[3] = p.k_off_2 * state[2]                            # unbinding of ligand #2 from state 3
        propensities[4] = p.k_p * ( state[0] + state[2] )                 # produce n1
        propensities[5] = p.k_f * state[0]                                # kpr forward step + GPCR event ligand #1
        propensities[6] = p.k_f * state[2]                                # kpr forward step + GPCR event ligand #2
        propensities[7] = p.k_off_1 * state[1]                            # unbinding of ligand #1 from 2
        propensities[8] = p.k_off_2 * state[3]                            # unbinding of ligand #2 from 4
        propensities[9] = p.k_p * ( state[1] + state[3] )                 # produce n2

    return propensities


def simulate_traj(num_steps=NUM_STEPS, init_cond=None, model=DEFAULT_MODEL, params=DEFAULT_PARAMS):
    """
    Simulate single trajectory for num_steps
    - model: model string
    - params: params object storing k_on, k_off, c etc
    """
    # prep trajectories
    times = np.zeros(num_steps)
    traj = np.zeros((num_steps, STATE_SIZE[model]))
    # setup init conds
    if init_cond is not None:
        try:
            assert len(init_cond) == STATE_SIZE[model]
        except AssertionError:
            print ("len(init_cond) = " , str(len(init_cond)))
            print ("STATE_SIZE[model] = " , str(STATE_SIZE[model]))
            assert len(init_cond) == STATE_SIZE[model]
    else:
        init_cond = INIT_CONDS[model]
    traj[0, :] = init_cond
    times[0] = 0.0
    for step in range(num_steps-1):
        # generate two U[0,1] random variables
        r1, r2 = np.random.random(2)
        # generate propensities and their partitions
        alpha = propensities(traj[step], model, params=params)
        alpha_partitions = np.zeros(len(alpha)+1)
        alpha_sum = 0.0
        for i in range(len(alpha)):
            alpha_sum += alpha[i]
            alpha_partitions[i + 1] = alpha_sum
        # find time to first reaction
        tau = np.log(1 / r1) / alpha_sum
        # pick a reaction
        r2_scaled = alpha_sum * r2
        for rxn_idx in range(len(alpha)):
            if alpha_partitions[rxn_idx] <= r2_scaled < alpha_partitions[rxn_idx + 1]:  # i.e. rxn_idx has occurred
                break
        # update state
        traj = update_state(traj, rxn_idx, step, model)
        times[step+1] = times[step] + tau
    return traj, times


def multitraj(num_traj, num_steps=NUM_STEPS, bound_probabilities=0.0, model=DEFAULT_MODEL, params=DEFAULT_PARAMS,
              init_cond_input=[]):
    """
    Return:
    - traj_array: num_steps x STATE_SIZE[model] x num_traj
    - times_array: num_steps x num_traj
    """
    traj_array = np.zeros((num_steps, STATE_SIZE[model], num_traj), dtype=int)
    times_array = np.zeros((num_steps, num_traj))
    # prep init cond of varying bound states (such that average I.C. matches bound_probabilities)
    if init_cond_input == []:
        init_cond_base = INIT_CONDS[model]
    else:
        init_cond_base = init_cond_input
    if type(bound_probabilities) == float: # for legacy purposes
        bound_probabilities = [1.0 - bound_probabilities, bound_probabilities]
    draws = np.random.choice(np.arange(0, RECEPTOR_STATE_SIZE[model]), size=num_traj, p=bound_probabilities)
    #draws = np.random.binomial(1, bound_probabilities, num_traj)
    # simulate k trajectories
    for k in range(num_traj):
        init_vector = np.zeros(RECEPTOR_STATE_SIZE[model])
        init_vector[draws[k]] = 1
        init_cond_base[:RECEPTOR_STATE_SIZE[model]-1] = init_vector[1:] # init_cond_base does not contain unbound state
        traj, times = simulate_traj(num_steps=num_steps, init_cond=init_cond_base, model=model, params=params)
        traj_array[:, :, k] = traj
        times_array[:, k] = times

    return traj_array, times_array


if __name__ == '__main__':
    # settings
    model = 'mode_1'
    num_traj = 20
    num_steps = 1000*2
    init_bound = 1.0
    # compute
    traj_array, times_array = multitraj(num_traj, bound_probabilities=init_bound, num_steps=num_steps, model=model)
    
    endtime = np.max(times_array[-1, :]) #for plotting analytic mean
    T = np.linspace(0, endtime, num_steps)
    amean = ns_mean(T)
    avar = ns_var(T)
    if np.min(avar) < 0:
        print('Warning: variance becomes negative due to numerical errors')
    
    avar1 = ns_mean(T) + np.sqrt(ns_var(T))
    avar2 = ns_mean(T) - np.sqrt(ns_var(T))
    
    
    # plot trajectories
    for k in range(num_traj):
        times_k = times_array[:, k]
        traj_k = traj_array[:, 1, k]
        plt.plot(times_k, traj_k, '--', lw=0.5, alpha=0.5)
    # decorate
    plt.plot(T,amean,'black', lw=0.8)
    plt.plot(T, avar1, 'black', linestyle='--', lw=0.8)
    plt.plot(T, avar2, 'black', linestyle='--', lw=0.8)
    plt.title('Model: %s - %d trajectories' % (model, num_traj))
    plt.xlabel('time')
    plt.ylabel(r'$\langle n_1 \rangle$')
#    plt.legend()
    plt.savefig('traj')
    plt.show()
    # T = np.linspace(0,10,100)
    # plt.plot(T,ns_var(T)/ns_mean(T)**2)
    # plt.ylim(0.05,.2)
    # plt.show()
    # plt.plot(T, ns_var(T))
    # plt.show()
    # plt.plot(T,ns_mean(T)**2)