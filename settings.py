import numpy as np
import os

"""
State for mode 1, 2: [bool, n]
    - bool == 1 implies BOUND, 0 implies UNBOUND
    - n is number of downstream molecules produced (due to being bound)

Tim: added option to use dimless parameters


TODO
- more comments
- fix or remove theory curves for 'direct'
- convert model structures into class
"""

# project level
FOLDER_OUTPUT = "output"
if not os.path.exists(FOLDER_OUTPUT):
    os.makedirs(FOLDER_OUTPUT)

# using dimless parameters
K = 1.0 #k_off/k_on 1
x = 0.1 #k_off/d_n 0.03
y = 10.1 #k_p/d_n 50.1
z = 5.1 #c/K 25.1

d_n = 1.0 #set this to get others
np.savez('gill_var', K=K, x=x, y=y,z=z,d_n=d_n)

# # data = np.load('gill_var.npz', allow_pickle=True)
# # x,y,z = data['x'], data['y'], data['z']
# # K, d_n = data['K'], data['d_n']
# # print(x,y,z,K,d_n)

# ##check the regime
# #ratio = (1+y+z+x*(1+z)**2)/((1+z)*(1+x+x*z))
# #print('Ratio of var:mean is',ratio)

# calculate dimful parameters, since everything else is in terms of these
GLOB_C = z*K
GLOB_K_OFF = x*d_n
GLOB_K_ON = GLOB_K_OFF/K
GLOB_K_P = y*d_n
GLOB_DEG_N = d_n
GLOB_DEG_M = 0.0 #last two not used
GLOB_K_F = 10.0

# # model parameters
# GLOB_C = 0.1
# GLOB_K_ON = 1.0
# GLOB_K_OFF = 0.01
# GLOB_K_P = 20.0
# GLOB_DEG_N = 0.0
# GLOB_DEG_M = 0.0
# GLOB_K_F = 10.0

# defined
GLOB_X = GLOB_C * GLOB_K_ON / GLOB_K_OFF
GLOB_PSS_BOUND = GLOB_X / (1 + GLOB_X)

# initial conditions (for single trajectory)
GLOB_N0 = 0.0
GLOB_M0 = 0.0
GLOB_BOUND_BOOL = 0
assert GLOB_BOUND_BOOL in [0, 1]

# misc
NUM_STEPS = 100

# models
VALID_MODELS = ['mode_1', 'mode_2', 'combined', 'kpr', 'two_ligand_kpr']
DEFAULT_MODEL = 'mode_1'

# model structures
NUM_RXN = {'mode_1': 4,
           'mode_2': 3,
           'combined': 5,
           'kpr': 7,
           'two_ligand_kpr': 10}
STATE_SIZE = {'mode_1': 2,
              'mode_2': 2,
              'combined': 3,
              'kpr': 4,
              'two_ligand_kpr': 8}
RECEPTOR_STATE_SIZE = {'mode_1': 2,
                       'mode_2': 2,
                       'combined': 2,
                       'kpr': 3,
                       'two_ligand_kpr': 5}

# init cond for each model
INIT_CONDS = {'mode_1': [GLOB_BOUND_BOOL, GLOB_N0],
              'mode_2': [GLOB_BOUND_BOOL, GLOB_M0],
              'combined': [GLOB_BOUND_BOOL, GLOB_N0, GLOB_M0],
              'kpr': [GLOB_BOUND_BOOL, 0, GLOB_N0, GLOB_M0],  # TODO handle init cond for kpr
              'two_ligand_kpr': [GLOB_BOUND_BOOL, 0, 0, 0, GLOB_N0, GLOB_M0, GLOB_N0, GLOB_M0]}
# reaction event update dictionary for each model
UPDATE_DICTS = {
    'mode_1': {0: np.array([1.0, 0.0]),  # binding
               1: np.array([-1.0, 0.0]),  # unbinding
               2: np.array([0.0, 1.0]),  # production
               3: np.array([0.0, -1.0])},  # degradation n
    'mode_2': {0: np.array([1.0, 1.0]),  # bind + GPCR event
               1: np.array([-1.0, 0.0]),  # unbind
               2: np.array([0.0, -1.0])},  # degradation m
    'combined': {0: np.array([1.0, 0.0, 1.0]),  # binding + GPCR event
                 1: np.array([-1.0, 0.0, 0.0]),  # unbinding
                 2: np.array([0.0, 1.0, 0.0]),  # production of n
                 3: np.array([0.0, -1.0, 0.0]),  # degradation n
                 4: np.array([0.0, 0.0, -1.0])},  # degradation m
    'kpr': {0: np.array([1.0, 0.0, 0.0, 0.0]),  # binding
            1: np.array([-1.0, 0.0, 0.0, 0.0]),  # unbinding
            2: np.array([-1.0, 1.0, 0.0, 1.0]),  # kpr forward step + GPCR event
            3: np.array([0.0, -1.0, 0.0, 0.0]),  # fall off
            4: np.array([0.0, 0.0, 1.0, 0.0]),  # produce n
            5: np.array([0.0, 0.0, -1.0, 0.0]),  # degradation n
            6: np.array([0.0, 0.0, 0.0, -1.0])},  # degradation m
    'two_ligand_kpr':
    #			     [  1,   2,   3,   4,  n1,  m1,  n2,  m2 ]
        {0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),   # ligand #1 binding
         1: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),   # ligand #2 binding
         2: np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # unbinding of ligand #1 from state 1
         3: np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # unbinding of ligand #2 from state 3
         4: np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),   # produce n1
         5: np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # kpr forward step + GPCR event ligand #1
         6: np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # kpr forward step + GPCR event ligand #2
         7: np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # unbinding of ligand #1 from 2
         8: np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),  # unbinding of ligand #2 from 4
         9: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])}   # produce n2

}
