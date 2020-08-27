#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:21:01 2020

@author: tim

Systematic plotting routine
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from numerics import Fisher, Fisher2, aFisher2, aInfo2, nInfo2 #Fisher function
from collections import OrderedDict
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

def P_normal(m,x,y,z):
    '''
    Likelihood P(m) as a normal distr.
    m: number or molecules
    x,y,z: non-dim parameters
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    return(np.sqrt(1/(2*np.pi*var))*np.exp(-(m-mean)**2/(2*var)))

'----------------------------------------------------------------'
#variables must be array or float or int for this to work
#numbers are kinda weird to avoid hitting poles of gamma function
l = int(1e2+1)
A = np.logspace(-2.1, 2.11, l) #x values
B =  np.logspace(-2.1, 2.11,l) #y values
C = 100  #z values

np.savez('variables',A=A,B=B,C=C) #save for future use
'----------------------------------------------------------------'
### Relative error in normal distr. approx. ###
A1,B1 = np.meshgrid(A,B)

#mesh of relative error, must check which vars are arrays
err = nInfo2(A1, B1, C)

np.save('err_norm',err)#_'+indep_name+dep_name, err)

##levels = MaxNLocator(nbins=5).tick_values(err.min(), 1) #setting contour levels
levels=[1e1,1e2,1e3,1e4]
levelsf = MaxNLocator(nbins=60).tick_values(err.min(), err.max()) #setting contourf levels

max = 1e6 #max color for colorbar

#mesh of regimes
R1_plot = R1(A1, B1, C)
levelscheck = np.linspace(R1_plot.min(), R1_plot.max(),7) #contours for regimes

CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A1, B1, err)#,norm=colors.LogNorm(vmin=err.min(), vmax=err.max()))#vmin=err.min(),vmax=max) #coloured plot
plt.colorbar()

CS = plt.contour(A1, B1, err, levels=levels, colors='white', linewidths=0.5) #only contours
plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True) #if manual=True, use command line for GUI to pop up
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$A$')
plt.ylabel('$B$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_norm')
plt.show()
plt.close()

### Plotting regime of validity ###
CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck)#region of validity
plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$A$')
plt.ylabel('$B$')
plt.savefig('regime')
plt.show()
plt.close()

### Analytic relative error ###
err= np.zeros([l,l])

for i in range(l): #must plot manually; numpy operations do not work with Fisher function
    if i%10==0:
        print(i, 'out of', l, 'rows') #just to see how fast it is
    for j in range(l):
        err[i,j] = aInfo2(A[j],B[i],C)

np.save('err_analytic',err)#_'+indep_name+dep_name, err)
#err = np.load('err_analytic.npy', allow_pickle=True)

CS1 = plt.contour(A1, B1, R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A1, B1, err)#,norm=colors.LogNorm(vmin=err.min(), vmax=err.max()))#vmin=err.min(),vmax=max) #coloured plot
plt.colorbar()

CS = plt.contour(A1, B1, err, levels=levels, colors='white', linewidths=0.5) #only contours
plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True) #if manual=True, use command line for GUI to pop up

plt.xlabel('$A$')
plt.ylabel('$B$')
plt.xscale('log')
plt.yscale('log')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_analytic')#,format='eps')
plt.show()
plt.close()
