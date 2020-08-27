'''
Tim: added steady-state degradation mean and variance to mode 1 expressions.
'''

import numpy as np
from numpy import sqrt, exp
import os

from params import DEFAULT_PARAMS
from settings import DEFAULT_MODEL, VALID_MODELS

#from mpi4py import MPI

def ns_mean(t, p=DEFAULT_PARAMS):
    '''
    Calculates mean for non-steady state mode1 w/ degradation. Used for testing in traj sim code.
    '''
    c, kon, koff, kp, x, pss, r, kd, d_m, k_f = p.unpack()
    fac1 = kd - np.exp(-(koff + c*kon)*t)*kd + (np.exp(-kd*t) - 1)*(koff + c*kon)
    fac2 = -c*kon*kp 
    den = kd*(koff + c*kon)*(-kd + koff + c*kon)
    return(fac1*fac2/den)

def ns_var(t, p=DEFAULT_PARAMS):
    '''
    Calculates variance for non-steady state mode1 w/ degradation. Used for testing in traj sim code.
    '''
    c, kon, koff, kp, x, pss, r, kd, d_m, k_f = p.unpack()
    
    fac1 = c*kon*kp
    
    # a = np.exp(2*kd*t)
    a1 = -c*kd*kon*(-2*kd + koff + c*kon)*(kd + koff + c*kon)*kp*np.exp(-2*(koff + c*kon)*t)
    a2_1 = -(2*kd - koff - c*kon)*(-kd + koff + c*kon)**2
    a2_2 = (koff + c*kon)*(kd + koff + c*kon) + koff*kp
    a2 = a2_1*a2_2
    a3_1 = np.exp(-(koff + c*kon)*t)*(kd - koff - c*kon)*(kd + koff + c*kon)
    a3_2 = kd*(2*kd - koff - c*kon)*(koff + c*kon) + 2*(kd - koff - c*kon)*(koff - c*kon)*kp
    a3 = a3_1*a3_2
    term1 = (a1+a2+a3)
    
    b = -((koff + c*kon)**2)
    b1 = -np.exp(-2*kd*t)*(kd - koff)*(kd + koff + c*kon)*kp
    b2_1 = (2*kd - koff - c*kon)
    b2_2 = np.exp(-kd*t)*(kd - koff - c*kon)*(kd + koff + c*kon) + np.exp((-kd - koff - c*kon)*t)*2*(kd - koff + c*kon)*kp
    b2 = b2_1*b2_2
    term2 = b*(b1+b2)
    
    num = fac1*(term1+term2)
    
    den = kd*(koff + c*kon)**2*(-2*kd + koff + c*kon)*(-kd + koff + 
   c*kon)**2*(kd + koff + c*kon)
    return(num/den)

def theory_moments(moment_times, bound_fraction, method='generating', init_n=0.0, init_m=0.0,
                   p=DEFAULT_PARAMS, model=DEFAULT_MODEL):
    """
    Returns, as a dict, multiple output timeseries (aligned to moment_times) depending on the model:
        - mode_1: mean(n)(t), var(n)(t), None, None, None
        - mode_2: None, None, mean(m)(t), var(m)(t), None
        - combined: mean(n)(t), var(n)(t), mean(m)(t), var(m)(t), cov(n,m)(t)
    """
    assert method in ["direct", "generating","degrade"]
    assert model in VALID_MODELS

    # unpack params
    c, k_on, k_off, k_p, x, pss, r, d_n, d_m, k_f = p.unpack()
#    assert d_n == 0.0  # no theory for this case
    assert d_m == 0.0  # no theory for this case
    init_p0 = 1 - bound_fraction
    # declare output
    theory_curves = {'mean_n': None,
                     'var_n': None,
                     'mean_m': None,
                     'var_m': None,
                     'cov_nm': None}

    if model == 'mode_1':
        theory_curves['mean_n'] = np.zeros(moment_times.shape[0])
        theory_curves['var_n'] = np.zeros(moment_times.shape[0])

        if method == "direct":
            delta1 = bound_fraction - pss
            C1 = pss ** 3 + 1 / r * (delta1 * pss + x / (1 + x) ** 3)
            C2 = pss ** 2 - delta1 * (1 - x) / (1 + x) ** 2
            C3 = delta1 * (1 / x) * (1 + x ** 2) / (1 + x) ** 2
            for idx, t in enumerate(moment_times):
                mean_n_val = k_p * pss * t + init_n + k_p * delta1 * (1 - np.exp(-r * t)) / r
                var_n_val = 2 * k_p ** 2 * (C1 * (t - 1 / r * (1 - np.exp(-r * t))) +
                                            C2 * t ** 2 / 2 +
                                            C3 * (1 - np.exp(-r * t) * (r * t + 1)) / r ** 2) + \
                            mean_n_val - mean_n_val ** 2
                theory_curves['mean_n'][idx] = mean_n_val
                theory_curves['var_n'][idx] = var_n_val
        elif method == 'generating':
            for idx, t in enumerate(moment_times):
                x1 = c * k_on * k_p * (np.exp(-r * t) - 1 + r * t) / r ** 2
                x2 = k_p * (k_off - np.exp(-r * t) * k_off + k_on * c * (r * t)) / r ** 2
                mean_n_val = x1 * (1 - bound_fraction) + x2 * bound_fraction

                var_term_1 = r ** 2 * ((1 - np.exp(-r * t)) * k_off * bound_fraction +
                                       c ** 2 * k_on ** 2 * (init_p0 + bound_fraction) * t +
                                       c * k_on * (k_off * bound_fraction * t + init_p0 * (
                                -1 + np.exp(-r * t) + k_off * t)))
                var_term_2 = k_p * ((1 - np.exp(-r * t)) * k_off * bound_fraction + c ** 2 * k_on ** 2 * (
                        init_p0 + bound_fraction) * t +
                                    c * k_on * (k_off * bound_fraction * t + init_p0 * (
                                -1 + np.exp(-r * t) + k_off * t))) ** 2
                var_term_3 = (c * k_on) ** 4 * (init_p0 + bound_fraction) * t ** 2 - 2 * k_off ** 2 * bound_fraction * (
                        np.exp(-r * t) - 1 + np.exp(-r * t) * k_off * t)
                var_term_4 = 2 * (k_on * c) ** 3 * t * (k_off * bound_fraction * t + init_p0 * (-1 + k_off * t))
                var_term_5 = 2 * c * k_off * k_on * (
                        init_p0 * (
                        2 * np.exp(-r * t) + k_off * t * np.exp(-r * t) + (-2 + k_off * t)) + bound_fraction * (
                                2 * np.exp(-r * t) - k_off * t * np.exp(-r * t) + 2 * (-1 + k_off * t)))
                var_term_6 = (k_on * c) ** 2 * (
                        k_off * bound_fraction * t * (4 + k_off * t) + init_p0 * (
                        -2 * np.exp(-r * t) + 2 * k_off * t * np.exp(-r * t) + (2 + (k_off * t) ** 2)))
                var_term_7 = var_term_3 + var_term_4 + var_term_5 + var_term_6

                var_n_val = (k_p / r ** 4) * (var_term_1 - var_term_2 + k_p * var_term_7)
                theory_curves['mean_n'][idx] = mean_n_val
                theory_curves['var_n'][idx] = var_n_val
        elif method == 'degrade_steady': #degregation, steady state
            z = k_off/d_n #non-dim parameters
            y = k_p/d_n
            for idx, t in enumerate(moment_times):
                mean_n_val = y * pss  #mean, var are constant but I kept loop so as to not break anything
                var_n_val = y**2*x*(x*z+1)/((1+x)*(1+z+x*z)) + y*pss - (y*pss)**2
                theory_curves['mean_n'][idx] = mean_n_val
                theory_curves['var_n'][idx] = var_n_val

        else:
            assert method == 'degrade' #degregation, not steady state
            z = k_off/d_n #non-dim parameters
            y = k_p/d_n
            k_d = d_n
            for idx, t in enumerate(moment_times):
                mean_n_val = ns_mean(t)
   #              num1 = -2*np.exp(-(k_d + k_off + c*k_on)*t)*k_d*k_off*k_p
   #              num2 = np.exp(-2*k_d*t)*(k_d + k_off +
   # c*k_on)*(np.exp(k_d*t)*(k_off + c*k_on)*(-k_d + k_off + c*k_on) + k_off*k_p)
   #              num3 = (k_d - k_off - c*k_on)*((k_off + c*k_on)*(k_d + k_off + c*k_on) + k_off*k_p)
   #              fac = c*k_on*k_p
   #              den  = k_d*(k_d - k_off - c*k_on)*(k_off + c*k_on)**2*(k_d + k_off + c*k_on)
   #              num = (num1+num2+num3)*fac
   #              var_n_val = num/den
                var_n_val = ns_var(t)
                # var_n_val = y**2*x*(x*z+1)/((1+x)*(1+z+x*z)) + y*pss - (y*pss)**2
                # mean_n_val = mean1*mean2/mean3
                theory_curves['mean_n'][idx] = mean_n_val
                theory_curves['var_n'][idx] = var_n_val

    elif model == 'mode_2':
        assert method == 'generating'
        theory_curves['mean_m'] = np.zeros(moment_times.shape[0])
        theory_curves['var_m'] = np.zeros(moment_times.shape[0])

        for idx, t in enumerate(moment_times):
            # mean
            x1 = k_on * c * (-k_off + np.exp(-r * t) * k_off - c * k_on + c * k_on * np.exp(-r * t))
            x2 = k_on * c * (k_on * c - k_on * c * np.exp(-r * t) + k_off ** 2 * t + c * k_off * k_on * t)
            # variance
            var_term_1 = c * k_on * (
                    -k_off + np.exp(-r * t) * k_off - c * k_on + c * k_on * np.exp(-r * t)) * bound_fraction / (
                                 r ** 2)
            var_term_2 = c * k_on * (c * k_on - c * k_on * np.exp(-r * t) + k_off ** 2 * t + c * k_off * k_on * t) / (
                    r ** 2)
            var_term_3 = c * k_on * (
                    -k_off + np.exp(-r * t) * k_off - c * k_on + c * np.exp(-r * t) * k_on) * bound_fraction / (
                                 r ** 2)
            var_term_4 = c * k_on * (c * k_on - c * k_on * np.exp(-r * t) + k_off ** 2 * t + c * k_off * k_on * t) / (
                    r ** 2)
            var_term_5 = k_off ** 3 * t ** 2 - 2 * c * k_on * (-1 + bound_fraction) * (
                    2 * np.exp(-r * t) + c * k_on * np.exp(-r * t) * t + (-2 + c * k_on * t))
            var_term_6 = k_off * (-2 * np.exp(-r * t) + 2 * np.exp(-r * t) * c * k_on * t - 4 * np.exp(
                -r * t) * bound_fraction * (1 + c * k_on * t) + (
                                          2 + (c * k_on * t) ** 2 + bound_fraction * (4 - 4 * c * k_on * t)))
            var_term_7 = 2 * k_off ** 2 * t * (
                    -bound_fraction * np.exp(-r * t) + (-1 - bound_fraction + c * k_on * t)) + var_term_6
            var_term_8 = (k_on * c) ** 2 * k_off / (r ** 4) * (var_term_5 + var_term_7)

            theory_curves['mean_m'][idx] = (x1 * bound_fraction + x2) / (r ** 2)
            theory_curves['var_m'][idx] = var_term_1 + var_term_2 - (var_term_3 + var_term_4) ** 2 + var_term_8

    elif model == 'combined':
        assert method == 'generating'
        theory_curves['mean_n'] = np.zeros(moment_times.shape[0])
        theory_curves['var_n'] = np.zeros(moment_times.shape[0])
        theory_curves['mean_m'] = np.zeros(moment_times.shape[0])
        theory_curves['var_m'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_nm'] = np.zeros(moment_times.shape[0])

        for idx, t in enumerate(moment_times):
            # mean n
            x1 = (1 - np.exp(-r * t)) * k_off
            x2 = (k_on * c) ** 2 * t + c * k_on * (
                    k_off * bound_fraction * t + init_p0 * (-1 + np.exp(-r * t) + k_off * t))
            # mean m
            y1 = k_on * c * np.exp(-r * t) * (-c * k_on + k_off * bound_fraction + c * k_on * bound_fraction) / (r ** 2)
            y2 = c * k_on * (
                    c * k_on - k_off * bound_fraction + k_off ** 2 * t + c * k_off * k_on * t - c * k_on * bound_fraction) / (
                         r ** 2)
            # variance n
            varN_val_1 = (np.exp(-2 * r * t) * k_p / (r ** 4)) * (
                    -c ** 2 * k_on ** 2 * k_p * init_p0 ** 2 + 2 * c * k_off * k_on * k_p * init_p0 * bound_fraction - k_off ** 2 * k_p * bound_fraction ** 2)
            varN_val_2 = varN_val_1 + (k_p / (r ** 4)) * (
                    -c * k_off ** 2 * k_on - 2 * c ** 2 * k_off * k_on ** 2 - c ** 3 * k_on ** 3 - 4 * c * k_off * k_on * k_p + c ** 2 * k_on ** 2 * k_p + k_off ** 3 * bound_fraction + 3 * c * k_off ** 2 * k_on * bound_fraction + 3 * c ** 2 * k_off * k_on ** 2 * bound_fraction)
            varN_val_25 = c ** 3 * k_on ** 3 * bound_fraction + 2 * k_off ** 2 * k_p * bound_fraction + 2 * c * k_off * k_on * k_p * bound_fraction - k_off ** 2 * k_p * bound_fraction ** 2
            varN_val_3 = varN_val_2 + (1 / r ** 4) * k_p * (
                    varN_val_25 - 2 * c * k_off * k_on * k_p * bound_fraction ** 2 - c ** 2 * k_on ** 2 * k_p * bound_fraction ** 2 + c * k_off ** 3 * k_on * t + 3 * c ** 2 * k_off ** 2 * k_on ** 2 * t + 3 * c ** 3 * k_off * k_on ** 3 * t + c ** 4 * k_on ** 4 * t + 2 * c * k_off ** 2 * k_on * k_p * t + 2 * c ** 2 * k_off * k_on ** 2 * k_p * t)
            varN_val_35 = 4 * c * k_off * k_on * k_p - 2 * c ** 2 * k_on ** 2 * k_p + c * k_off ** 2 * k_on * init_p0 + 2 * c ** 2 * k_off * k_on ** 2 * init_p0 + c ** 3 * k_on ** 3 * init_p0
            varN_val_36 = 2 * c ** 2 * k_on ** 2 * k_p * init_p0 - k_off ** 3 * bound_fraction - 2 * c * k_off ** 2 * k_on * bound_fraction - c ** 2 * k_off * k_on ** 2 * bound_fraction - 2 * k_off ** 2 * k_p * bound_fraction
            varN_val_37 = -2 * c * k_off * k_on * k_p * bound_fraction + 2 * c ** 2 * k_on ** 2 * k_p * bound_fraction - 2 * c * k_off * k_on * k_p * init_p0 * bound_fraction - 2 * c ** 2 * k_on ** 2 * k_p * init_p0 * bound_fraction
            varN_val_38 = 2 * k_off ** 2 * k_p * bound_fraction ** 2 + 2 * c * k_off * k_on * k_p * bound_fraction ** 2 + 2 * c * k_off ** 2 * k_on * k_p * t + 2 * c ** 2 * k_off * k_on ** 2 * k_p * t - 2 * c ** 2 * k_off * k_on ** 2 * k_p * init_p0 * t
            varN_val_4 = varN_val_3 + (np.exp(-r * t) * k_p / (r ** 4)) * (
                    varN_val_35 + varN_val_36 + varN_val_37 + varN_val_38 - 2 * c ** 3 * k_on ** 3 * k_p * init_p0 * t - 2 * k_off ** 3 * k_p * bound_fraction * t - 2 * c * k_off ** 2 * k_on * k_p * bound_fraction * t)
            # variance m
            varM_val_1 = np.exp(-r * t) * r ** 2 * (c * k_on * (-1 + bound_fraction) + k_off * bound_fraction)
            varM_val_2 = c * k_on * (c * k_on - k_off * bound_fraction - c * k_on * bound_fraction + np.exp(-r * t) * (
                    c * k_on * (
                    -1 + bound_fraction) + k_off * bound_fraction) + k_off ** 2 * t + c * k_off * k_on * t) ** 2
            varM_val_3 = r ** 2 * (k_off * (-bound_fraction + k_off * t) + c * k_on * (1 - bound_fraction + k_off * t))
            varM_val_4 = k_off ** 3 * t ** 2 - 2 * c * k_on * (-1 + bound_fraction) * (
                    2 * np.exp(-r * t) + c * k_on * t * np.exp(-r * t) + (-2 + c * k_on * t))
            varM_val_5 = 2 * k_off ** 2 * t * (-bound_fraction * np.exp(-r * t) + (-1 - bound_fraction + c * k_on * t))
            varM_val_6 = k_off * ((-2 + 2 * c * k_on * t - 4 * bound_fraction * (1 + c * k_on * t)) * np.exp(
                -r * t) + 2 + c ** 2 * k_on ** 2 * t ** 2 + bound_fraction * (4 - 4 * c * k_on * t))
            # covariance n,m
            cov_val_1 = -c ** 2 * k_on ** 2 + 2 * c * k_off * k_on * bound_fraction + 2 * c ** 2 * k_on ** 2 * bound_fraction - k_off ** 2 * bound_fraction ** 2 - 2 * c * k_off * k_on * bound_fraction ** 2 - c ** 2 * k_on ** 2 * bound_fraction ** 2
            cov_val_2 = k_off ** 2 - 4 * c * k_off * k_on + 3 * k_off ** 2 * bound_fraction + 4 * c * k_off * k_on * bound_fraction + c ** 2 * k_on ** 2 * bound_fraction - k_off ** 2 * bound_fraction ** 2 - 2 * c * k_off * k_on * bound_fraction ** 2 - c ** 2 * k_on ** 2 * bound_fraction ** 2 - k_off ** 3 * t + c ** 2 * k_off * k_on ** 2 * t
            cov_val_3 = -k_off ** 2 + 4 * c * k_off * k_on + c ** 2 * k_on ** 2 - 3 * k_off ** 2 * bound_fraction - 6 * c * k_off * k_on * bound_fraction - 3 * c ** 2 * k_on ** 2 * bound_fraction + 2 * k_off ** 2 * bound_fraction ** 2 + 4 * c * k_off * k_on * bound_fraction ** 2
            cov_val_4 = 2 * c ** 2 * k_on ** 2 * bound_fraction ** 2 + 3 * c * k_off ** 2 * k_on * t + 2 * c ** 2 * k_off * k_on ** 2 * t - c ** 3 * k_on ** 3 * t - 3 * k_off ** 3 * bound_fraction * t - 5 * c * k_off ** 2 * k_on * bound_fraction * t - c ** 2 * k_off * k_on ** 2 * bound_fraction * t + c ** 3 * k_on ** 3 * bound_fraction * t

            theory_curves['mean_n'][idx] = k_p / (r ** 2) * (x1 * bound_fraction + x2)
            theory_curves['var_n'][idx] = varN_val_4
            theory_curves['mean_m'][idx] = y1 + y2
            theory_curves['var_m'][idx] = c * k_on / (r ** 4) * (
                    varM_val_1 - varM_val_2 + varM_val_3 + c * k_off * k_on * (
                    varM_val_4 + varM_val_5 + varM_val_6))
            theory_curves['cov_nm'][idx] = -c * k_on * k_p * np.exp(-2 * r * t) / (r ** 4) * (
                cov_val_1) - c * k_on * k_p / (r ** 4) * (cov_val_2) - c * k_on * k_p * np.exp(-r * t) / (r ** 4) * (
                                                   cov_val_3 + cov_val_4)

    elif model == 'two_ligand_kpr':
        c1 = p.c1
        c2 = p.c2
        k_off_1 = p.k_off_1
        k_off_2 = p.k_off_2

        # Steady state receptor occupancies
        D1 = (k_off_1 * k_off_2 + c2 * k_off_1 * k_on + c1 * k_off_2 * k_on)
        P0 = k_off_1 * k_off_2 / D1
        P1 = c1 * k_off_1 * k_off_2 * k_on / ((k_f + k_off_1) * D1)
        P2 = c1 * k_f * k_off_2 * k_on / ((k_f + k_off_1) * D1)
        P3 = c2 * k_off_1 * k_off_2 * k_on / ((k_f + k_off_2) * D1)
        P4 = 1 - P0 - P1 - P2 - P3

        theory_curves['mean_n1'] = np.zeros(moment_times.shape[0])
        theory_curves['var_n1'] = np.zeros(moment_times.shape[0])
        theory_curves['mean_n2'] = np.zeros(moment_times.shape[0])
        theory_curves['var_n2'] = np.zeros(moment_times.shape[0])

        theory_curves['mean_m1'] = np.zeros(moment_times.shape[0])
        theory_curves['var_m1'] = np.zeros(moment_times.shape[0])
        theory_curves['mean_m2'] = np.zeros(moment_times.shape[0])
        theory_curves['var_m2'] = np.zeros(moment_times.shape[0])

        theory_curves['cov_n1m1'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_n1m2'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_n2m1'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_n2m2'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_n1n2'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_m1m2'] = np.zeros(moment_times.shape[0])

        import os
        covariance_fnames = ["CovN1N1.txt", "CovN1M1.txt", "CovN1N2.txt", "CovN1M2.txt", "CovM1M1.txt",
                             "CovN2M1.txt", "CovM1M2.txt", "CovN2N2.txt", "CovN2M2.txt", "CovM2M2.txt"]
        theory_expression_strings = {'var_n1':"", 'cov_n1m1':"", 'cov_n1n2':"", 'cov_n1m2':"", 'var_m1':"",
                                'cov_n2m1':"", 'cov_m1m2':"", 'var_n2':"", 'cov_n2m2':"", 'var_m2':""}
        theory_expression_strings_ordered_keys = ['var_n1', 'cov_n1m1', 'cov_n1n2', 'cov_n1m2', 'var_m1',
                                                  'cov_n2m1', 'cov_m1m2', 'var_n2', 'cov_n2m2', 'var_m2']
        for i in range(10):
            with open(os.path.join(os.getcwd(), "CovarianceExpressions", covariance_fnames[i])) as f:
                expression = f.read().replace("Sqrt", "sqrt")
                theory_expression_strings[theory_expression_strings_ordered_keys[i]] = expression

        for idx, t in enumerate(moment_times):
            theory_curves['mean_n1'][idx] = k_p * t * (P1 + P3)
            theory_curves['mean_m1'][idx] = k_on * (c1 + c2) * t * P0
            theory_curves['mean_n2'][idx] = k_p * t * (P2 + P4)
            theory_curves['mean_m2'][idx] = (k_f * t) * (P1 + P3)

            # From Mathematica
            x1 = k_on * c1 / k_off_1
            g1 = k_off_1 / k_f
            x2 = k_on * c2 / k_off_2
            g2 = k_off_2 / k_f

            theory_curves['var_n1'][idx] = eval(theory_expression_strings['var_n1'])

                #k_p * t*((c1 * k_off_1 * k_off_2 * k_on) / ((k_f + k_off_1)*(c1 * k_off_2 * k_on + k_off_1 * (k_off_2 + c2 * k_on))) +
                #    (c2 * k_off_1 * k_off_2 * k_on) / ((k_f + k_off_2)*(c1 * k_off_2 * k_on + k_off_1 * (k_off_2 + c2 * k_on))) -
                #    (k_off_1**2 * k_off_2**2 * (c2 * (k_f + k_off_1) + c1 * (k_f + k_off_2))**2 * k_on**2 * k_p * t) /
                #    ((k_f + k_off_1)**2 * (k_f + k_off_2)**2 * (c1 * k_off_2 * k_on + k_off_1 * (k_off_2 + c2 * k_on))**2) +
                #    (k_p * t * (g1 * (1 + g2) * x1 + (1 + g1) * g2 *x2)**2) / ((1 + g1)**2 * (1 + g2)**2 * (1 + x1 + x2)**2) +
                #    (1 / (g1 * (1 + g1)**3 * g2 * (1 + g2)**3 * k_f * (1 + x1 + x2)**3)) * 2 *k_p * (g2**3 * (1 + g2) * x1 * x2**2 + g1**4 * x2 *(
                #        (1 + g2)*(g2**2 + g2 * (-1 + x1) * x1 + x1**2) + g2 * (g2 + x1 + 2 * g2 * x1) * x2 + g2**2 * x2**2) +
                #    g1**2 * g2 * ((1 + g2)**3 * x1 * (1 + x1 + x1**2) + (1 + g2)*(x1 + g2 * (3 + x1 * (5 - g2 + 2*(2 + g2) * x1))) * x2 +
                #                  (3 * g2 + (2 + 4 * g2 + g2**3) * x1) * x2**2 + 3 * g2 * x2**3) + g1**3 * (g2*(1 + g2)**3 * x1 +
                #                                                                                          (1 + g2)*(x1**2 + g2**2 * (3 + 4 * x1)) * x2 +
                #                                                                                          3 * g2 * (g2 + x1 + 2 *g2 * x1) * x2**2 + 3 *g2**2 * x2**3) +
                #   g1 * g2**2 * x2 * (1 + x1 + 2 *x1**2 + x2 + x2**2 + g2**2 * x1 * (-1 + x1 + 2 * x2) + g2(1 + x1 * (3 * x1 + x2)))))

            theory_curves['var_n2'][idx] = eval(theory_expression_strings['var_n2'])
            theory_curves['var_m1'][idx] = eval(theory_expression_strings['var_m1'])
            theory_curves['var_m2'][idx] = eval(theory_expression_strings['var_m2'])
            theory_curves['cov_n1m1'][idx] = eval(theory_expression_strings['cov_n1m1'])
            theory_curves['cov_n1m2'][idx] = eval(theory_expression_strings['cov_n1m2'])
            theory_curves['cov_n1n2'][idx] = eval(theory_expression_strings['cov_n1n2'])
            theory_curves['cov_n2m1'][idx] = eval(theory_expression_strings['cov_n2m1'])
            theory_curves['cov_n2m2'][idx] = eval(theory_expression_strings['cov_n2m2'])
            theory_curves['cov_m1m2'][idx] = eval(theory_expression_strings['cov_m1m2'])


    else:
        assert method == 'generating'
        assert model == 'kpr'
        theory_curves['mean_n'] = np.zeros(moment_times.shape[0])
        theory_curves['var_n'] = np.zeros(moment_times.shape[0])
        theory_curves['mean_m'] = np.zeros(moment_times.shape[0])
        theory_curves['var_m'] = np.zeros(moment_times.shape[0])
        theory_curves['cov_nm'] = np.zeros(moment_times.shape[0])

        g = k_off / k_f
        for idx, t in enumerate(moment_times):
            theory_curves['mean_n'][idx] = (k_f * k_p * t) / (k_f + k_off) * x / (1 + x)
            theory_curves['mean_m'][idx] = (k_f * k_off * t) / (k_f + k_off) * x / (1 + x)

            A1 = k_p * x * ((1 + g) ** 4 * k_off ** 2 * (1 + x) ** 4) ** (-1.0)
            A2 = (1 + g) ** 3 * k_off ** 2 * t * (1 + x) ** 3
            A30 = -1 + k_off * t * (1 + x)
            A31 = g ** 3 * (-1 + k_off * t) * (1 + x) ** 3
            A32 = g * (3 + x) * A30
            A33 = g ** 2 * A30 * (3 + x * (3 + x))
            theory_curves['var_n'][idx] = A1 * (A2 + 2 * k_p * (A30 + A31 + A32 + A33))

            A = x / ((1 + g) ** 4 * (1 + x) ** 4)
            B = (1 + g) * k_off * t * (1 + x) * (1 + 2 * g + x ** 2 + g ** 2 * (1 + x) ** 2)
            C = 2 * x * (1 + g * (3 + x + g * (3 + x * (3 + x))))
            theory_curves['var_m'][idx] = A * (B + C)

            A = k_p * x / ((1 + g) ** 4 * (1 + x) ** 4 * k_off)
            B = -1 + x + g ** 3 * (-1 + k_off * t) * (1 + x) ** 3
            C = k_off * (t - t * x ** 2) - g * (-1 + x) * (3 + x) * (-1 + k_off * t * (1 + x))
            D = g ** 2 * (-3 + x ** 2 * (2 + x) + k_off * t * (1 + x) * (3 + x))
            theory_curves['cov_nm'][idx] = A * (B + C + D)
    return theory_curves


def est_x_from_n(n, params, t):
    return n / (params.k_p * t - n)


def est_c_from_n(n, params, t):
#    return n / (params.k_p * t - n) * (params.k_off / params.k_on) #no degradation
    if params.k_p/params.d_n - n == 0.0:
        print('Oops')

    return n / (params.k_p/params.d_n - n) * (params.k_off / params.k_on) #degredation


def est_k_off_from_n(n, params, t):
    return (params.k_p * t - n) / n * (params.k_on * params.c)


def est_x_from_m(m, params, t):
    # TODO issue that this depends on k_off? x contains k_off info...
    print ("warning... inferring x = k_on c / k_off using information about k_off see formulae.py, est_x_from_m(...)")
    return m / (params.k_off * t - m)


def est_c_from_m(m, params, t):
    return m / (params.k_off * t - m) * (params.k_off / params.k_on)


def est_k_off_from_m(m, params, t):
    k_on_c = params.k_on * params.c
    return k_on_c * m / (k_on_c * t - m)


def est_k_d_from_nm(n, m, params, t):
    return (m / n) * (params.k_p / params.k_on)  # note k_d = k_off / k_on


def est_k_off_from_nm(n, m, params, t):
    if n == 0:
        print ('WARNING for est_k_off_from_nm() -- n==0 at time %.3f (set n=1)') % t
        n = 1.0
    return (float(m) / float(n)) * (params.k_p)


def est_c_from_nm(n, m, params, t):
    # k_off_guess = est_k_off_from_nm(n, m, params, t)
    # return (k_off_guess / params.k_on) * float(n) / (params.k_p * t - float(n))
    c_star = (1.0 / params.k_on) * (m / t) / (1 - n / (params.k_p * t))
    return c_star


def estimate_general(state, params, t, model, est):
    assert model in VALID_MODELS
    assert est in ['x', 'c', 'k_off']
    if model == 'mode_1':
        if est == 'x':
            est = est_x_from_n(state[1], params, t)
        elif est == 'c':
            est = est_c_from_n(state[1], params, t)
        else:
            est = est_k_off_from_n(state[1], params, t)
    elif model == 'mode_2':
        if est == 'x':
            est = est_x_from_m(state[1], params, t)
        elif est == 'c':
            est = est_c_from_m(state[1], params, t)
        else:
            est = est_k_off_from_m(state[1], params, t)
    elif model == 'combined':
        if est == 'x':
            print ("Warning, est == 'x' for model 'combined' not implemented, setting est = -99")
            est = -99
        elif est == 'c':
            est = est_c_from_nm(state[1], state[2], params, t)
        else:
            est = est_k_off_from_nm(state[1], state[2], params, t)
    elif model == 'kpr':
        if est == 'x':
            print ("Warning, est == 'x' for model 'combined' not implemented, setting est = -99")
            est = -99
        elif est == 'c':
            g = params.k_off / params.k_f
            est = est_c_from_nm(state[1] * (1 + g), state[2], params, t)
        else:
            est = est_k_off_from_nm(state[1], state[2], params, t)
    else:
        print ("Warning, model %s not implemented in estimate_general() in formulae.py") % model
        assert 1 == 2
    return est
