#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:25:49 2020

@author: tim
PLotting non-steady plots w/ parallelization
"""

import numpy as np
from numerics import Fisher_ns
import matplotlib.pyplot as plt
from time import time

import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes


l = 100 #array length

T = np.logspace(-2, 1.3, l)

x = np.logspace(-2, 1.1, l)

y = 5.1
z = 5.1

np.savez('variables',T=T,y=y,z=z,x=x)

Tgrid, xgrid = np.meshgrid(T,y)
err = np.zeros([l,l])

size = l
block_size = 5

result = np.ctypeslib.as_ctypes(np.zeros((size, size)))
shared_array = sharedctypes.RawArray(result._type_, result)


def fill_per_window(args):
    window_x, window_y = args
    tmp = np.ctypeslib.as_array(shared_array)
    start = time()
    print('Block started:',args)
    for idx_x in range(window_x, window_x + block_size):
        for idx_y in range(window_y, window_y + block_size):
            tmp[idx_x, idx_y] = Fisher_ns(T[idx_y],x[idx_x],y, z, estimate='spec')
    end = time()
    print('Block completed:',args)
    print('Time taken:', end-start)

window_idxs = [(i, j) for i, j in
               itertools.product(range(0, size, block_size),
                                 range(0, size, block_size))]
start_t = time()



p = Pool(25)
res = p.map(fill_per_window, window_idxs)
result = np.ctypeslib.as_array(shared_array)
end_t = time()

print('Total time:', (end_t-start_t)/60,'min')
np.save('err_dynamical',result)

# plt.pcolormesh(Tgrid, xgrid, result,vmin=result.min(),vmax=result.max()) #coloured plot
# plt.colorbar()
# print(np.array_equal(X, result))


# for i in range(l):
#     start = time()
#     for j in range(l):
#         if j%5 ==0:
#             print('column',j)
#         err[i,j] = Fisher_ns(T[j],x[i],y,z)
#     end = time()
#     print('row',i, 'took', end-start,'s')

# np.save('err_dynamical',err)
# plt.pcolormesh(Tgrid, xgrid, err,vmin=err.min(),vmax=err.max()) #coloured plot
# plt.colorbar()
