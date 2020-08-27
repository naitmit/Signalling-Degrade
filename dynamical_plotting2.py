#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting heatmaps in dynamical case, using parallel processing.
Algorithm borrowed from https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html.
"""

import numpy as np
from numerics import Fisher_ns, Fisher_nsg_num
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
# err = np.zeros([l,l])

size = l
block_size = 5

result = np.ctypeslib.as_ctypes(np.zeros((size, size))) #for sharring array memory
shared_array = sharedctypes.RawArray(result._type_, result)


def fill_per_window(args): #fills array in findows of nxn blocks
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)
    start = time()
    print('Block started:',args)
    for idx_x in range(window_x, window_x + block_size):
        for idx_y in range(window_y, window_y + block_size):
            tmp[idx_x, idx_y] = Fisher_nsg_num(T[idx_y],x[idx_x],y,z) #this is for the normal approx
    end = time()
    print('Block completed:',args)
    print('Time taken:', end-start)

window_idxs = [(i, j) for i, j in
                itertools.product(range(0, size, block_size),
                                  range(0, size, block_size))] #indices for windows
start_t = time()
p = Pool(4) #number of processes
res = p.map(fill_per_window, window_idxs)
result = np.ctypeslib.as_array(shared_array)
end_t = time()

print('Total time:', (end_t-start_t)/60,'min')
np.save('err_dynamical_norm_ty',result)


'-----------Plotting------------'
result = np.load('err_dynamical_norm_ty.npy', allow_pickle=True)
result1 = np.load('err_dynamical.npy', allow_pickle=True)

levels=[1e-2,1e-1,1e0]#,1e0,1.2,1e4]

plt.pcolormesh(Tgrid, xgrid, result1, norm=colors.LogNorm(vmin=result1.min(), vmax=result1.max())) #coloured plot
plt.colorbar()
plt.xlabel('T')
plt.ylabel('y')
CS = plt.contour(Tgrid, xgrid, result1, levels=levels, colors='black', linewidths=0.5)
plt.savefig('err_analytic_ty')
plt.show()

plt.pcolormesh(Tgrid, xgrid, result, norm=colors.LogNorm(vmin=result.min(), vmax=result.max())) #coloured plot
plt.colorbar()
plt.xlabel('T')
plt.ylabel('y')
plt.savefig('err_norm_ty')
plt.show()

diff = np.abs((result-result1)/result1)
plt.pcolormesh(Tgrid, xgrid, diff, norm=colors.LogNorm(vmin=diff.min(), vmax=diff.max())) #coloured plot
plt.colorbar()
plt.xlabel('T')
plt.ylabel('y')
plt.savefig('err_diff_ty')
plt.show()
