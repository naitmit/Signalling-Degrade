#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:21:01 2020

@author: tim

Only plots A, B, C given the saved variable and accuracy arrays (edited version of 
plot_analysis).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors

#define functions
def R1(A,B,C):
    '''
    Ratio to check for normal distr: <n>/std.dev
    x,y,z: non-dim parameters
    '''
    x = B*C
    y = C
    z = A/B
    num = y*z*(1+x+x*z)
    den = 1+y+z+x*(1+z)**2
    return(np.sqrt(num/den))

def rel_e(A,B,C):
    '''
    Relative error in z estimate
    x,y,z: non-dim parameters
    '''
    x = B*C
    y = C
    z = A/B
    term1 = y*z*(1 + x + x*z)/((1 + z)**2*(1 + y + z + x*(1 + z)**2))
    term2 = (1 + y + z - y*z + x**2*(1 + z)**3 -
  x*(1 + z)*(-2*(1 + z) + y*(-1 + 2*z)))**2/(2*(1 + z)**2*(1 + x +
   x*z)**2*(1 + y + z + x*(1 + z)**2)**2)
    err = 1/(term1+term2)
    return(err)
    
def rel_e2(A,B,C):
    fac1 = 1/C
    fac2 = (B + A)/A
    fac3 = 1 + 2*A/B + (A/B)**2 + 2/B
    return(fac1*fac2*fac3)

'----------------------------------------------------------------'

tf = input("Manually pick contour? Input [0] for 'no', [1] for'yes':")
tf = int(tf)

data = np.load('variables.npz', allow_pickle=True)
A,B,C = data['A'], data['B'], data['C']


'----------------------------------------------------------------'
### Relative error in normal distr. approx. ###
A1,B1 = np.meshgrid(A,B)

err_n = np.load('err_norm.npy', allow_pickle=True)
# err_n = err_n[:-1, :-1]

##levels = MaxNLocator(nbins=5).tick_values(err.min(), 1) #setting contour levels
levels=[1e-1,1e0,1e1,1e2,1e3,1e4]
levelsf = MaxNLocator(nbins=60).tick_values(err_n.min(), err_n.max()) #setting contourf levels

max = 1e6 #max color for colorbar

#mesh of regimes
R1_plot = R1(A1, B1, C)
levelscheck =[1.0,2.0,3.0]# np.linspace(R1_plot.min(), R1_plot.max(),4) #contours for regimes

CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A1, B1, err_n,norm=colors.LogNorm(vmin=err_n.min(), vmax=err_n.max()))#vmin=err.min(),vmax=max) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

CS = plt.contour(A1, B1, err_n, levels=levels, colors='white', linewidths=0.5) #only contours
# plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f')#, manual=True) #if manual=True, use command line for GUI to pop up

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f', manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f')

plt.xlabel('$A$')
plt.ylabel('$B$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_norm')
plt.show()
plt.close()

### Plotting regime of validity ###
CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck)#region of validity
plt.xscale('log')
plt.yscale('log')
plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True)

plt.xlabel('$A$')
plt.ylabel('$B$')
plt.savefig('regime')
plt.show()
plt.close()

### Analytic relative error ###
err_a = np.load('err_analytic.npy', allow_pickle=True)
# err_a = err_a[:-1, :-1]

CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A1, B1, err_a,norm=colors.LogNorm(vmin=err_a.min(), vmax=err_a.max()))#vmin=err.min(),vmax=max) #coloured plot
plt.colorbar()

CS = plt.contour(A1, B1, err_a, levels=levels, colors='white', linewidths=0.5) #only contours
# plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f')#, manual=True) #if manual=True, use command line for GUI to pop up
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f', manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f')

plt.xlabel('$A$')
plt.ylabel('$B$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_analytic')#,format='eps')
plt.show()
plt.close()

### Differences in error ###
derr = np.abs(err_a - err_n)/err_a
# err_a = err_a[:-1, :-1]
plt.pcolormesh(A1, B1, derr,norm=colors.LogNorm(vmin=derr.min(), vmax=derr.max()))#vmin=err.min(),vmax=max) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
# plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')#, manual=True)

if tf == 1:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f', manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')
# CS = plt.contour(A1, B1, derr, levels=levels, colors='white', linewidths=0.5) #only contours
# plt.clabel(CS, CS.levels, inline=True, fontsize=10, fmt = '%1.0f')#, manual=True) #if manual=True, use command line for GUI to pop up

plt.xlabel('$A$')
plt.ylabel('$B$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('diff')#,format='eps')
plt.show()
plt.close()
