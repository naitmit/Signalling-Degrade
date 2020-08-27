'''
Tim: edited execution loop, only using mode 1 w/ degradation
'''

import matplotlib.pyplot as plt
import numpy as np

from formulae import theory_moments, estimate_general
from params import Params, DEFAULT_PARAMS
from settings import GLOB_PSS_BOUND
from trajectory_plotting import plot_traj_and_mean_sd, plot_means, plot_vars, plot_hist, plot_estimation
from trajectory_simulate import multitraj

from time import time
#from mpi4py import MPI

#plt.style.use('parameters.mplstyle')


def get_state_at_t(traj, times, t, last_step=0):
    step = last_step
    while times[step] <= t:
        step += 1
    return traj[step - 1, :], step - 1


def get_moment_timeseries(traj_array, times_array, params):
    """
    Returns, as a dict, multiple output timeseries (aligned to moment_times) depending on the model:
        - mode_1: mean(n)(t), var(n)(t), distribution(n)(t), estimate_x(t)
        - mode_2: mean(m)(t), var(m)(t), distribution(m)(t), estimate_x(t)
        - combined: mean(n)(t), var(n)(t), mean(m)(t), var(m)(t), cov(n,m)(t), distribution(n)(t), distribution(m)(t),
                    estimate_c(t), estimate_k_off(t),
    """
    # prepare moment times
    num_traj = np.shape(traj_array)[-1]
    dt = np.mean(times_array[1, :])
    endtime = np.min(times_array[-1, :])
    moment_times = np.arange(0.0, endtime, dt)
    # pass previous step to get_state_at_t(...) to speedup
    last_step = np.zeros(num_traj, dtype=int)
    # prepare value dict
    moment_curves = {'mean_n': None,
                     'var_n': None,
                     'mean_m': None,
                     'var_m': None,
                     'cov_nm': None,
                     'distribution_n': None,
                     'distribution_m': None,
                     'estimate_x': None,
                     'estimate_c': None,
                     'estimate_k_off': None}

    if model in ['mode_1', 'mode_2']:
        distro_1or2 = {'mode_1': 'distribution_n', 'mode_2': 'distribution_m'}[model]
        mean_1or2 = {'mode_1': 'mean_n', 'mode_2': 'mean_m'}[model]
        var_1or2 = {'mode_1': 'var_n', 'mode_2': 'var_m'}[model]
        moment_curves[mean_1or2] = np.zeros(len(moment_times))
        moment_curves[var_1or2] = np.zeros(len(moment_times))
        moment_curves[distro_1or2] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['estimate_x'] = np.zeros((len(moment_times), num_traj))
        moment_curves['estimate_c'] = np.zeros((len(moment_times), num_traj))
        moment_curves['estimate_k_off'] = np.zeros((len(moment_times), num_traj))

        for idx, t in enumerate(moment_times):
            statesum = 0.0
            statesquaresum = 0.0
            for k in range(num_traj):
                state_at_t, step = get_state_at_t(traj_array[:, :, k], times_array[:, k], t, last_step=last_step[k])
                last_step[k] = step
                statesum += state_at_t[1]
                statesquaresum += state_at_t[1]**2
                # store n(t) and m(t) for each trajectory to get histogram evolution
                moment_curves[distro_1or2][idx][k] = state_at_t[1]
                # estimate x, c, k_off from mode_1 and mode_2 current "data" n(t) or m(t)
#                moment_curves['estimate_x'][idx][k] = estimate_general(state_at_t, params, t, model, 'x')
#                moment_curves['estimate_c'][idx][k] = estimate_general(state_at_t, params(), t, model, 'c') # TODO: uncomment when estimating
#                moment_curves['estimate_k_off'][idx][k] = estimate_general(state_at_t, params, t, model, 'k_off')

            moment_curves[mean_1or2][idx] = statesum / num_traj
            moment_curves[var_1or2][idx] = statesquaresum / num_traj - moment_curves[mean_1or2][idx]**2
    elif model == 'combined':
        moment_curves = {key: np.zeros(len(moment_times)) for key in ['mean_n', 'mean_m', 'var_n', 'var_m', 'cov_nm']}
        moment_curves['distribution_n'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['distribution_m'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['estimate_c'] = np.zeros((len(moment_times), num_traj))
        moment_curves['estimate_k_off'] = np.zeros((len(moment_times), num_traj))

        for idx, t in enumerate(moment_times):
            statesum_n = 0.0
            statesquaresum_n = 0.0
            statesum_m = 0.0
            statesquaresum_m = 0.0
            stateprod_nm = 0.0
            for k in range(num_traj):
                state_at_t, step = get_state_at_t(traj_array[:, :, k], times_array[:, k], t, last_step=last_step[k])
                last_step[k] = step
                statesum_n += state_at_t[1]
                statesquaresum_n += state_at_t[1] ** 2
                statesum_m += state_at_t[2]
                statesquaresum_m += state_at_t[2] ** 2
                stateprod_nm += state_at_t[1] * state_at_t[2]
                # store n(t) and m(t) for each trajectory to get histogram evolution
                moment_curves['distribution_n'][idx][k] = state_at_t[1]
                moment_curves['distribution_m'][idx][k] = state_at_t[2]
                # estimate x from mode_1 and mode_2 current "data" n(t) or m(t)
                moment_curves['estimate_c'][idx][k] = estimate_general(state_at_t, params, t, model, 'c')
                moment_curves['estimate_k_off'][idx][k] = estimate_general(state_at_t, params, t, model, 'k_off')

            moment_curves['mean_n'][idx] = statesum_n / num_traj
            moment_curves['mean_m'][idx] = statesum_m / num_traj
            moment_curves['var_n'][idx] = statesquaresum_n / num_traj - moment_curves['mean_n'][idx]**2
            moment_curves['var_m'][idx] = statesquaresum_m / num_traj - moment_curves['mean_m'][idx]**2
            moment_curves['cov_nm'][idx] = stateprod_nm / num_traj - \
                                           moment_curves['mean_n'][idx] * moment_curves['mean_m'][idx]

    elif model == 'two_ligand_kpr':
        moment_curves = {key: np.zeros(len(moment_times)) for key in ['mean_n1', 'mean_m1', 'var_n1', 'var_m1', 'cov_n1m1',
                                                                      'mean_n2', 'mean_m2', 'var_n2', 'var_m2', 'cov_n2m2',
                                                                      'cov_n1m2', 'cov_n1n2', 'cov_n2m1', 'cov_m1m2']}
        moment_curves['distribution_n1'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['distribution_m1'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['distribution_n2'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['distribution_m2'] = np.zeros((len(moment_times), num_traj), dtype=int)

        moment_curves['estimate_c'] = np.zeros((len(moment_times), num_traj))
        moment_curves['estimate_k_off'] = np.zeros((len(moment_times), num_traj))

        for idx, t in enumerate(moment_times):
            statesum_n1 = 0.0
            statesum_n2 = 0.0
            statesquaresum_n1 = 0.0
            statesquaresum_n2 = 0.0
            statesum_m1 = 0.0
            statesum_m2 = 0.0
            statesquaresum_m1 = 0.0
            statesquaresum_m2 = 0.0
            stateprod_n1m1 = 0.0
            stateprod_n1m2 = 0.0
            stateprod_n2m1 = 0.0
            stateprod_n2m2 = 0.0
            stateprod_n1n2 = 0.0
            stateprod_m1m2 = 0.0

            for k in range(num_traj):
                state_at_t, step = get_state_at_t(traj_array[:, :, k], times_array[:, k], t, last_step=last_step[k])

                n1_at_t = state_at_t[4];
                n2_at_t = state_at_t[6];
                m1_at_t = state_at_t[5];
                m2_at_t = state_at_t[7];

                last_step[k] = step
                statesum_n1 += n1_at_t
                statesum_n2 += n2_at_t
                statesquaresum_n1 += n1_at_t ** 2
                statesquaresum_n2 += n2_at_t ** 2
                statesum_m1 += m1_at_t
                statesum_m2 += m2_at_t
                statesquaresum_m1 += m1_at_t ** 2
                statesquaresum_m2 += m2_at_t ** 2
                stateprod_n1m1 += n1_at_t * m1_at_t
                stateprod_n1m2 += n1_at_t * m2_at_t
                stateprod_n2m1 += n2_at_t * m1_at_t
                stateprod_n2m2 += n2_at_t * m2_at_t
                stateprod_n1n2 += n1_at_t * n2_at_t
                stateprod_m1m2 += m1_at_t * m2_at_t

                # store n(t) and m(t) for each trajectory to get histogram evolution
                moment_curves['distribution_n1'][idx][k] = n1_at_t
                moment_curves['distribution_m1'][idx][k] = m1_at_t
                moment_curves['distribution_n2'][idx][k] = n2_at_t
                moment_curves['distribution_m2'][idx][k] = m2_at_t

                # estimate x from mode_1 and mode_2 current "data" n(t) or m(t)
                #moment_curves['estimate_c'][idx][k] = estimate_general(state_at_t, params, t, model, 'c')
                #moment_curves['estimate_k_off'][idx][k] = estimate_general(state_at_t, params, t, model, 'k_off')

            moment_curves['mean_n1'][idx] = statesum_n1 / num_traj
            moment_curves['mean_m1'][idx] = statesum_m1 / num_traj
            moment_curves['mean_n2'][idx] = statesum_n2 / num_traj
            moment_curves['mean_m2'][idx] = statesum_m2 / num_traj
            moment_curves['var_n1'][idx] = statesquaresum_n1 / num_traj - moment_curves['mean_n1'][idx] ** 2
            moment_curves['var_m1'][idx] = statesquaresum_m1 / num_traj - moment_curves['mean_m1'][idx] ** 2
            moment_curves['var_n2'][idx] = statesquaresum_n2 / num_traj - moment_curves['mean_n2'][idx] ** 2
            moment_curves['var_m2'][idx] = statesquaresum_m2 / num_traj - moment_curves['mean_m2'][idx] ** 2
            moment_curves['cov_n1m1'][idx] = stateprod_n1m1 / num_traj - \
                                           moment_curves['mean_n1'][idx] * moment_curves['mean_m1'][idx]
            moment_curves['cov_n1m2'][idx] = stateprod_n1m2 / num_traj - \
                                           moment_curves['mean_n1'][idx] * moment_curves['mean_m2'][idx]
            moment_curves['cov_n2m1'][idx] = stateprod_n2m1 / num_traj - \
                                           moment_curves['mean_n2'][idx] * moment_curves['mean_m1'][idx]
            moment_curves['cov_n2m2'][idx] = stateprod_n2m2 / num_traj - \
                                           moment_curves['mean_n2'][idx] * moment_curves['mean_m2'][idx]
            moment_curves['cov_n1n2'][idx] = stateprod_n1n2 / num_traj - \
                                           moment_curves['mean_n1'][idx] * moment_curves['mean_n2'][idx]
            moment_curves['cov_m1m2'][idx] = stateprod_m1m2 / num_traj - \
                                           moment_curves['mean_m1'][idx] * moment_curves['mean_m2'][idx]

    else:
        assert model == 'kpr'
        moment_curves = {key: np.zeros(len(moment_times)) for key in ['mean_n', 'mean_m', 'var_n', 'var_m', 'cov_nm']}
        moment_curves['distribution_n'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['distribution_m'] = np.zeros((len(moment_times), num_traj), dtype=int)
        moment_curves['estimate_c'] = np.zeros((len(moment_times), num_traj))
        moment_curves['estimate_k_off'] = np.zeros((len(moment_times), num_traj))

        for idx, t in enumerate(moment_times):
            statesum_n = 0.0
            statesquaresum_n = 0.0
            statesum_m = 0.0
            statesquaresum_m = 0.0
            stateprod_nm = 0.0
            for k in range(num_traj):
                state_at_t, step = get_state_at_t(traj_array[:, :, k], times_array[:, k], t, last_step=last_step[k])
                last_step[k] = step
                statesum_n += state_at_t[2]
                statesquaresum_n += state_at_t[2] ** 2
                statesum_m += state_at_t[3]
                statesquaresum_m += state_at_t[3] ** 2
                stateprod_nm += state_at_t[2] * state_at_t[3]
                # store n(t) and m(t) for each trajectory to get histogram evolution
                moment_curves['distribution_n'][idx][k] = state_at_t[2]
                moment_curves['distribution_m'][idx][k] = state_at_t[3]
                # estimate x from mode_1 and mode_2 current "data" n(t) or m(t)
#                moment_curves['estimate_c'][idx][k] = estimate_general(state_at_t, params, t, model, 'c')
#                moment_curves['estimate_k_off'][idx][k] = estimate_general(state_at_t, params, t, model, 'k_off')

            moment_curves['mean_n'][idx] = statesum_n / num_traj
            moment_curves['mean_m'][idx] = statesum_m / num_traj
            moment_curves['var_n'][idx] = statesquaresum_n / num_traj - moment_curves['mean_n'][idx] ** 2
            moment_curves['var_m'][idx] = statesquaresum_m / num_traj - moment_curves['mean_m'][idx] ** 2
            moment_curves['cov_nm'][idx] = stateprod_nm / num_traj - \
                                           moment_curves['mean_n'][idx] * moment_curves['mean_m'][idx]
    return moment_curves, moment_times


if __name__ == '__main__':
    start = time()
    # settings
    model = 'mode_1'
    # model = 'mode_2'
    #model = 'combined'
    #model = 'kpr'
    #model = 'two_ligand_kpr'

    #for model in ['mode_1', 'mode_2', 'combined', 'kpr']:
    for model in [model]:
        print(model)
        num_traj = 1000*4
        num_steps = 1000*4
        init_bound = 0 #DEFAULT_PARAMS.pss
        # model specification
        #params = DEFAULT_PARAMS
#        params = Params(k_on=1.0, k_p=20, k_f=1.0)
#        params.c1 = 10.0
#        params.c2 = 4.0
#        params.k_off_1 = 2.0
#        params.k_off_2 = 1.0
#
#        # Steady state receptor occupancies
#        D1 = (params.k_off_1 * params.k_off_2 + params.c2 * params.k_off_1 * params.k_on + params.c1 * params.k_off_2 * params.k_on)
#        P0 = params.k_off_1 * params.k_off_2 / D1
#        P1 = params.c1 * params.k_off_1 * params.k_off_2 * params.k_on / ((params.k_f + params.k_off_1) * D1)
#        P2 = params.c1 * params.k_f * params.k_off_2 * params.k_on / ((params.k_f + params.k_off_1) * D1)
#        P3 = params.c2 * params.k_off_1 * params.k_off_2 * params.k_on / ((params.k_f + params.k_off_2) * D1)
#        P4 = 1 - P0 - P1 - P2 - P3

        # simulate trajectories
        #   default for bound_probabilities is to just input init_bound
        init_bound = 0.0
        traj_array, times_array = multitraj(num_traj, bound_probabilities=init_bound, num_steps=num_steps, model=model)
        # compute moments from data
        simdata, moment_times = get_moment_timeseries(traj_array, times_array, params=Params)
        # expectations from theory
        theory_curves = theory_moments(moment_times, init_bound, method="degrade", model=model)


#        #compare without degradation
#        params1 = DEFAULT_PARAMS
#        params1.d_n = 0.0
#        traj_array1, times_array1 = multitraj(num_traj, bound_probabilities=init_bound, num_steps=num_steps, model=model)
#        # compute moments from data
#        simdata1, moment_times1 = get_moment_timeseries(traj_array1, times_array1, params=params1)
#        # expectations from theory
#        theory_curves1 = theory_moments(moment_times1, init_bound, method="degrade", model=model)

        # specify histogram timepoints
        num_points = 60
        l=len(moment_times)
        # hist_steps = [int(i*num_steps/num_points)+1 for i in range(num_points)]
        hist_steps = np.arange(1,l,np.ceil(l/num_points),int)
        # hist_steps = [-1]

        # model dependent plotting
        if model == 'mode_1':
            """
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='n', state_idx=1,
                                  data_mean=simdata['mean_n'], data_var=simdata['var_n'],
                                  theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'],
                                  title='%s <n>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_n'], model, theory_mean=theory_curves['mean_n'], state_label='n',
                       title='%s <n>(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_n'], model, theory_var=theory_curves['var_n'], state_label='n',
                      title='%s Var(n)(t) for %d trajectories' % (model, num_traj))
            plot_estimation(moment_times, simdata['estimate_x'], model, params, theory_curves, theory=True, est='x')
            plot_estimation(moment_times, simdata['estimate_c'], model, params, theory_curves, est='c')
            plot_estimation(moment_times, simdata['estimate_k_off'], model, params, theory_curves, est='k_off')
            """
#            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='n', state_idx=1,
#                                  data_mean=simdata['mean_n'], data_var=simdata['var_n'],
#                                  theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'],
#                                  title='%s <n>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_n'], model, theory_mean=theory_curves['mean_n'], state_label='n',
                       title='%s <n>(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_n'], model, theory_var=theory_curves['var_n'], state_label='n',
                      title='%s Var(n)(t) for %d trajectories' % (model, num_traj))

            for step in hist_steps:
                plot_hist(moment_times, simdata['distribution_n'], step, model, state_label='n',
                          theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'], show=True)
#            plot_estimation(moment_times, simdata['estimate_c'], model, Params(), theory_curves, est='c')

        elif model == 'mode_2':
            """
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='m', state_idx=1,
                                  data_mean=simdata['mean_m'], data_var=simdata['var_m'],
                                  theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'],
                                  title='%s <m>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_m'], model, theory_mean=theory_curves['mean_m'], state_label='m',
                       title='%s <m>(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_m'], model, theory_var=theory_curves['var_m'], state_label='m',
                      title='%s Var(m)(t) for %d trajectories' % (model, num_traj))
            plot_estimation(moment_times, simdata['estimate_x'], model, params, theory_curves, est='x')
            plot_estimation(moment_times, simdata['estimate_c'], model, params, theory_curves, est='c')
            plot_estimation(moment_times, simdata['estimate_k_off'], model, params, theory_curves, est='k_off')
            """
            for step in hist_steps:
                plot_hist(moment_times, simdata['distribution_m'], step, model, state_label='m',
                          theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'], show=False)

        elif model == 'combined':
            """
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='n', state_idx=1,
                                  data_mean=simdata['mean_n'], data_var=simdata['var_n'],
                                  theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'],
                                  title='%s <n>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='m', state_idx=2,
                                  data_mean=simdata['mean_m'], data_var=simdata['var_m'],
                                  theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'],
                                  title='%s <m>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_n'], model, theory_mean=theory_curves['mean_n'], state_label='n',
                       title='%s <n>(t) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_m'], model, theory_mean=theory_curves['mean_m'], state_label='m',
                       title='%s <m>(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_n'], model, theory_var=theory_curves['var_n'], state_label='n',
                      title='%s Var(n)(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_m'], model, theory_var=theory_curves['var_m'], state_label='m',
                      title='%s Var(m)(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['cov_nm'], model, theory_var=theory_curves['cov_nm'], state_label='nm',
                      title='%s Cov(n,m)(t) for %d trajectories' % (model, num_traj))
            plot_estimation(moment_times, simdata['estimate_c'], model, params, theory_curves, theory=True, est='c')
            plot_estimation(moment_times, simdata['estimate_k_off'], model, params, theory_curves, theory=True, est='k_off')
            """
            for step in hist_steps:
                plot_hist(moment_times, simdata['distribution_n'], step, model, state_label='n',
                          theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'], show=False)
                plot_hist(moment_times, simdata['distribution_m'], step, model, state_label='m',
                          theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'], show=False)

        elif model == 'two_ligand_kpr':
            plot_hist_flag = False
            plot_cov_traj_flag = True
            if plot_hist_flag == True:
                for step in hist_steps:
                    plot_hist(moment_times, simdata['distribution_n1'], step, model, state_label='n1',
                              theory_mean=None, theory_var=None, show=False)
                    plot_hist(moment_times, simdata['distribution_m1'], step, model, state_label='m1',
                              theory_mean=None, theory_var=None, show=False)
                    plot_hist(moment_times, simdata['distribution_n2'], step, model, state_label='n2',
                              theory_mean=None, theory_var=None, show=False)
                    plot_hist(moment_times, simdata['distribution_m2'], step, model, state_label='m2',
                              theory_mean=None, theory_var=None, show=False)
            if plot_cov_traj_flag == True:
                plot_vars(moment_times, simdata['var_n1'], model, theory_var=theory_curves['var_n1'],
                          title='Two Ligand Var N1', state_label='n1', show=False)
                plt.close()
                plot_vars(moment_times, simdata['var_n2'], model, theory_var=theory_curves['var_n2'],
                          title='Two Ligand Var N2', state_label='n2', show=False)
                plt.close()
                plot_vars(moment_times, simdata['var_m1'], model, theory_var=theory_curves['var_m1'],
                          title='Two Ligand Var M1', state_label='m1', show=False)
                plt.close()
                plot_vars(moment_times, simdata['var_m2'], model, theory_var=theory_curves['var_m2'],
                          title='Two Ligand Var M2', state_label='m2', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_n1m1'], model, theory_var=theory_curves['cov_n1m1'],
                          title='Two Ligand Cov N1M1', state_label='N1M1', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_n1m2'], model, theory_var=theory_curves['cov_n1m2'],
                          title='Two Ligand Cov N1M2', state_label='N1M2', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_n2m1'], model, theory_var=theory_curves['cov_n2m1'],
                          title='Two Ligand Cov N2M1', state_label='N2M1', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_n2m2'], model, theory_var=theory_curves['cov_n2m2'],
                          title='Two Ligand Cov N2M2', state_label='N2M2', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_n1n2'], model, theory_var=theory_curves['cov_n1n2'],
                          title='Two Ligand Cov N1N2', state_label='N1N2', show=False)
                plt.close()
                plot_vars(moment_times, simdata['cov_m1m2'], model, theory_var=theory_curves['cov_m1m2'],
                          title='Two Ligand Cov M1M2', state_label='M1M2', show=False)
                plt.close()

        else:
            assert model == 'kpr'
            """
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='n', state_idx=2,
                                  data_mean=simdata['mean_n'], data_var=simdata['var_n'],
                                  theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'],
                                  title='%s <n>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, state_label='m', state_idx=3,
                                  data_mean=simdata['mean_m'], data_var=simdata['var_m'],
                                  theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'],
                                  title='%s <m>(t) +- sqrt(var(t)) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_n'], model, theory_mean=theory_curves['mean_n'], state_label='n',
                       title='%s <n>(t) for %d trajectories' % (model, num_traj))
            plot_means(moment_times, simdata['mean_m'], model, theory_mean=theory_curves['mean_m'], state_label='m',
                       title='%s <m>(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_n'], model, theory_var=theory_curves['var_n'], state_label='n',
                      title='%s Var(n)(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['var_m'], model, theory_var=theory_curves['var_m'], state_label='m',
                      title='%s Var(m)(t) for %d trajectories' % (model, num_traj))
            plot_vars(moment_times, simdata['cov_nm'], model, theory_var=theory_curves['cov_nm'], state_label='nm',
                      title='%s Cov(n,m)(t) for %d trajectories' % (model, num_traj))
            """
            for step in hist_steps:
                plot_hist(moment_times, simdata['distribution_n'], step, model, state_label='n',
                          theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'], show=False)
                plot_hist(moment_times, simdata['distribution_m'], step, model, state_label='m',
                          theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'], show=False)
            # TODO
            """
            #plot_estimation(moment_times, simdata['estimate_c'], model, params, theory_curves, theory=True, est='c')
            #plot_estimation(moment_times, simdata['estimate_k_off'], model, params, theory_curves, theory=True, est='k_off')
            for step in hist_steps:
                plot_hist(moment_times, simdata['distribution_n'], step, model, state_label='n',
                          theory_mean=theory_curves['mean_n'], theory_var=theory_curves['var_n'], show=False)
                plot_hist(moment_times, simdata['distribution_m'], step, model, state_label='m',
                          theory_mean=theory_curves['mean_m'], theory_var=theory_curves['var_m'], show=False)
            """
    end = time()
    run = end-start
    print('Run time was', run,'s')
