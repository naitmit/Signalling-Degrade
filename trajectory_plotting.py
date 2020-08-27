'''
Tim: edited plot_hist, added true P(n) as function
'''

import matplotlib.pyplot as plt
import numpy as np
import os
# import scipy.special as sp

from laguerre_testing import P_mp, P_ns #exact prob distr

from settings import FOLDER_OUTPUT
variables = np.load('gill_var.npz', allow_pickle=True) #load the variables
x, y, z = variables['x'], variables['y'], variables['z']

vP_mp = np.vectorize(P_mp) #vectorize prob
vP_ns = np.vectorize(P_ns)

def plot_traj_and_mean_sd(traj_array, times_array, moment_times, model, data_mean=None, data_var=None, theory_mean=None,
                          theory_var=None, state_label='n', state_idx=1, title='', show=True):
    # plot individual trajectories
    num_traj = traj_array.shape[-1]
    for k in range(num_traj):
        times_k = times_array[:, k]
        traj_k = traj_array[:, state_idx, k]
        plt.plot(times_k, traj_k, '--', lw=0.5, alpha=0.5)
    # plot trajectory moments
    if data_mean is not None:
        plt.plot(moment_times, data_mean, 'k', lw=2, label="Data")
        if data_var is not None:
            plt.plot(moment_times, data_mean + np.sqrt(data_var), '--k', lw=2)
            plt.plot(moment_times, data_mean - np.sqrt(data_var), '--k', lw=2)
    # plot theory moments (generating function)
    if theory_mean is not None:
        plt.plot(moment_times, theory_mean, 'r', lw=2, label="Analytic")
        if theory_var is not None:
            plt.plot(moment_times, theory_mean + np.sqrt(theory_var), '--r', lw=2)
            plt.plot(moment_times, theory_mean - np.sqrt(theory_var), '--r', lw=2)
    # decorate
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel(state_label)
    plt.legend()
    plt.savefig(FOLDER_OUTPUT + os.sep + 'traj_%s_%s.png' % (model, state_label))
    if show:
        plt.show()
    return plt.gca()


def plot_means(moment_times, data_mean, model, theory_mean=None, title='', state_label='n', show=True):
    # plt.scatter(moment_times, data_mean, s=4.0, c='k', marker='s', label="data", alpha=0.5)
    # plt.scatter(moment_times, theory_mean, s=4.0, c='r', marker='s', label="steady-state degradation", alpha=0.5)
    plt.plot(moment_times, data_mean, '--k', lw=2, label="Data")
    plt.plot(moment_times, theory_mean, '--r', lw=2, label="Analytic")
    # plt.title(title)
    plt.xlabel('Time(s)')
    # plt.ylabel('<%s>(t)' % state_label)
    plt.ylabel('Mean')
    plt.legend()
    plt.savefig(FOLDER_OUTPUT + os.sep + 'traj_%s_mean_%s.png' % (model, state_label))
    if show:
        plt.show()
    return plt.gca()


def plot_vars(moment_times, data_var, model, theory_var=None, title='', state_label='n', show=True, save=True):
    plt.plot(moment_times, data_var, '--k', lw=2, label="Data")
    if theory_var is not None:
        plt.plot(moment_times, theory_var, '--r', lw=2, label="Analytic")
    # plt.title(title)
    plt.xlabel('Time(s)')
    # plt.ylabel('Var(%s)' % state_label)
    plt.ylabel('Variance')
    plt.legend()
    if save:
        plt.savefig(FOLDER_OUTPUT + os.sep + 'traj_%s_var_%s.png' % (model, state_label))
    if show:
        plt.show()
    return plt.gca()


def plot_hist(moment_times, data_observations, step, model, state_label='n', theory_mean=None, theory_var=None, show=True):
    time = moment_times[step]
    hist_data = data_observations[step, :]
    #hist_bins = np.arange(0, hist_data.max() + 1.5) - 0.5
    hist_bins = np.linspace(0,hist_data.max(), 50)
    points = np.linspace(0, hist_data.max(), 1000)
    num_traj = len(hist_data)
    if theory_mean is not None:
        assert theory_var is not None
        mu = theory_mean[step]
        sigma = np.sqrt(theory_var[step])
        a1 = max(0,np.floor(mu-3.5*sigma)) #bounds for graph domains
        b1 = np.ceil(mu+3.5*sigma)
        hist_bins = np.arange(0, b1, 1)#np.ceil(7*sigma/50)) #ceil instead of int to get rid of 0s
#        points = np.linspace(mu-3.5*sigma, mu+3.5*sigma, 1000)
        points = np.arange(0, b1, 1, int)
        normal_at_bins = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (points - mu)**2 / (2 * sigma**2))
        actual_p = vP_mp(points,x,y,z)
        if y <= 12: #in this case, not a lot of molecules - can calculate prob w/ function
            prob_ns = vP_ns(points,time) #dynamical prob distr.
            plt.plot(points, prob_ns, linestyle='-', color='orange', label='Analytic Non-Steady')
        plt.plot(points, actual_p, '--r', label='Analytic Steady State')
        plt.plot(points, normal_at_bins, '--b', label='Normal Non-Steady')
    count, bins, _ = plt.hist(data_observations[step, :]+0.01, bins=hist_bins, color="#3F5D7D", ec='k', density = True)  # TODO cleanup
    plt.title(r'Time:%.2f s' % (time))
    # plt.title('Histogram of Gillespie Algorithm (%d traj) \n Time %f' % (num_traj, time))
    plt.xlabel(r'%s' %(state_label))
    plt.ylabel(r'Probability')
    plt.legend()
#    plt.ylim(0,0.02)

    # sum1 = np.sum(count) #checking if they all sum to 1
    # sum2 = np.sum(actual_p)
    # print('Histogram sums to', sum1, 'Predicted sums to', sum2)

    # plt.savefig(FOLDER_OUTPUT + os.sep + 'hist_%s_%sf_%d.png' % (model, state_label, step))
    # plt.title('Histogram (%d traj) for %s, %s at step:%d, time:%.2f' % (num_traj, model, state_label, step, time))
    plt.savefig(FOLDER_OUTPUT + os.sep + 'hist_%s_%s_%d_%.2f.png' % (model, state_label, step, time))
    if show:
        plt.show()
    plt.close()
    return plt.gca()
#    plt.show()
#    plt.plot(points, actual_p, '--r')
#    count, bins, _ = plt.hist(data_observations[step, :]+0.01, bins=hist_bins, color="#3F5D7D", ec='k', density = True)  # TODO cleanup
#    #plt.title(r'Histogram of Gillespie Algorithm (%d traj), time:%.2f' % (num_traj, time))
#    plt.title('Histogram of Gillespie Algorithm (%d traj) \n Time %f' % (num_traj, time))
#    plt.xlabel(r'%s' %(state_label))
#    plt.ylabel(r'probability')
#    plt.show()

def plot_estimation(moment_times, estimate_data, model, params, theory_curves, est='x', theory=False, show=True):
    # TODO care normalize var and implement theory
    print ("CARE NORMALIZE EST VAR IN PLOT")
    p = params

    def true_model_value():
        if est == 'x':
            model_value = p.k_on * p.c / p.k_off
        elif est == 'c':
            model_value = p.c
        else:
            assert est == 'k_off'
            model_value = p.k_off
        return model_value

    def est_theory(idx, t):
        if model == 'mode_1':
            assert est == 'x'
            mu_n = theory_curves['mean_n'][idx]
            x_star = mu_n/(p.k_p*t - mu_n)
            est_mu_t = x_star
            est_var_t = (1/mu_n) * ((1 + x_star)**2 + 2 * p.k_p / p.k_off)

        elif model == 'mode_2':
            assert est == 'x'
            print ("WARNING, est_theory() in plot_estimation() not implemented for mode_2")
            assert 1 == 2
            est_mu_t = -1
            est_var_t = -1

        else:
            assert model == 'combined'
            assert est in ['c', 'k_off']
            mu_n = theory_curves['mean_n'][idx]
            mu_m = theory_curves['mean_m'][idx]
            """
            k_d_star = (p.k_p / p.k_on) * mu_m / mu_n  # this is "k_d_star = k_off_star / p.k_on"
            c_star = k_d_star * mu_n / (p.k_p * t - mu_n)
            """
            k_off_star = p.k_p * mu_m / mu_n
            #c_star = (k_off_star / p.k_on) * mu_n / (p.k_p * t - mu_n)
            c_star = (1.0 / p.k_on) * (mu_m / t) / (1 - mu_n / (p.k_p *t))
            x_star = c_star * p.k_on / k_off_star

            # terms common to both sigma_c and s0gma_k_off
            factor_1 = 1 + x_star + x_star ** 2 + x_star ** 3
            factor_2 = 1 + 2 * p.k_p / (k_off_star * (1 + x_star) ** 2)
            factor_3 = 1 - k_off_star / (p.k_p + k_off_star * (1 + x_star) ** 2)
            den = p.k_p * t * x_star
            if est == 'c':
                est_mu_t = c_star
                est_var_t = c_star ** 2 * factor_1 * factor_2 * factor_3 / den
            else:
                est_mu_t = k_off_star
                factor_4 = (1 + p.k_p / k_off_star) / (x_star ** 2 + p.k_p / k_off_star)
                est_var_t = k_off_star ** 2 * factor_1 * factor_2 * factor_3 * factor_4 / den
        return est_mu_t, est_var_t

    estimate_mean_at_t = np.zeros(moment_times.shape)
    estimate_var_at_t = np.zeros(moment_times.shape)
    model_value = true_model_value()
    for idx in range(len(moment_times)):
        estimate_mean_at_t[idx] = np.mean(estimate_data[idx, :])
        estimate_var_at_t[idx] = np.var(estimate_data[idx, :])

    if theory:
        theory_mean_at_t = np.zeros(moment_times.shape)
        theory_var_at_t = np.zeros(moment_times.shape)
        for idx in range(len(moment_times)):
            est_mu_t, est_var_t = est_theory(idx, moment_times[idx])
            theory_mean_at_t[idx] = est_mu_t
            theory_var_at_t[idx] = est_var_t

    plt.close()
    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    plt.suptitle('Estimation performance over time (%d traj)' % estimate_data.shape[-1])
    # plot mean over time
    ax1 = fig.add_subplot(121)
    ax1.scatter(moment_times, estimate_mean_at_t, s=1, label='sample mean of estimate')
    ax1.plot(moment_times, [model_value for _ in moment_times], '--k', label='true value')
    if theory:
        ax1.plot(moment_times, theory_mean_at_t, '--b', label='theory')
    ax1.set_title('mean of estimate for %s' % est)
    ax1.set_xlabel('t')
    ax1.set_ylabel('<%s_guess>' % est)
    ax1.legend()
    # plot var over time
    ax2 = fig.add_subplot(122)
    ax2.scatter(moment_times, estimate_var_at_t, s=1, label='sample var of estimate')
    if theory:
        ax2.plot(moment_times, theory_var_at_t, '--b', label='theory')
    ax2.set_title('variance of estimate for %s' % est)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Var(%s_guess)' % est)
    ax2.legend()
    # save and show
    plt.savefig(FOLDER_OUTPUT + os.sep + 'est_%s_%s.png' % (model, est))
    if show:
        plt.show()
    return plt.gca()

# if __name__ == '__main__':
    # m = np.arange(0,50,1)
    # plt.plot(m,P(m))
    # plt.ylim(0,0.02)
    