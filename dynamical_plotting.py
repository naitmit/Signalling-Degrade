#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:25:49 2020

Plotting non-steady 2D heatmaps given the saved arrays in .npy files. Same as dynamical_plotting2 except data is already generated.
err_dynamical_norm_tx.npy: error with normal approx
err_dynamical.npy: true error

"""

import numpy as np
from numerics import Fisher_ns, Fisher_nsg_num
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterSciNotation
from time import time

import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes


l = 100 #array length

T = np.logspace(-2, 1.3, l)

x = np.logspace(-2, 1.1, l)

y = 5.1
z = 5.1


Tgrid, xgrid = np.meshgrid(T,x)

tf = input("Manually pick contour? Input [0] for 'no', [1] for'yes':")
tf = int(tf)
# err = np.zeros([l,l])

# size = l
# block_size = 5

# result = np.ctypeslib.as_ctypes(np.zeros((size, size)))
# shared_array = sharedctypes.RawArray(result._type_, result)


# def fill_per_window(args):
#     window_x, window_y = args
#     tmp = np.ctypeslib.as_array(shared_array)
#     start = time()
#     print('Block started:',args)
#     for idx_x in range(window_x, window_x + block_size):
#         for idx_y in range(window_y, window_y + block_size):
#             tmp[idx_x, idx_y] = Fisher_nsg_num(T[idx_y],x[idx_x],y,z)
#     end = time()
#     print('Block completed:',args)
#     print('Time taken:', end-start)

# window_idxs = [(i, j) for i, j in
#                 itertools.product(range(0, size, block_size),
#                                   range(0, size, block_size))]
# start_t = time()
# p = Pool(4)
# res = p.map(fill_per_window, window_idxs)
# result = np.ctypeslib.as_array(shared_array)
# end_t = time()

# print('Total time:', (end_t-start_t)/60,'min')
# np.save('err_dynamical_norm',result)

result = np.load('err_dynamical_norm_tx.npy', allow_pickle=True)
result1 = np.load('err_dynamical.npy', allow_pickle=True)

levels=[0.5e1,1e2,1e3,1e4]#,1e0,1.2,1e4]

fmt = LogFormatterSciNotation()
fmt.create_dummy_axis()

plt.pcolormesh(Tgrid, xgrid, result1, norm=colors.LogNorm(vmin=result1.min(), vmax=result1.max())) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('T')
plt.ylabel('x')
CS = plt.contour(Tgrid, xgrid, result1, levels=levels, colors='white', linewidths=0.5)

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt, manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt)

plt.savefig('err_analytic_tx')
plt.show()

plt.pcolormesh(Tgrid, xgrid, result, norm=colors.LogNorm(vmin=result.min(), vmax=result.max())) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

CS = plt.contour(Tgrid, xgrid, result, levels=levels, colors='white', linewidths=0.5)

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt, manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt)
plt.xlabel('T')
plt.ylabel('x')
plt.savefig('err_norm_tx')
plt.show()

diff = np.abs((result-result1)/result1)
plt.pcolormesh(Tgrid, xgrid, diff, norm=colors.LogNorm(vmin=diff.min(), vmax=diff.max())) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt, manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt)
plt.xlabel('T')
plt.ylabel('x')
plt.savefig('err_diff_tx')
plt.show()
