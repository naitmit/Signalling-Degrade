#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:21:01 2020

@author: tim

For plotting two of (x,y,z) with given saved variables and calculated error files; edited version of.
For plotting in other paramters, see plot_newvar_analysis
"""
import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation
import matplotlib.colors as colors
from numerics import Fisher, rel_e #Fisher function
from collections import OrderedDict

#define functions
def R1(x,y,z):
    '''
    Ratio to check for normal distr: <n>/std.dev
    x,y,z: non-dim parameters
    '''
    num = y*z*(1+x+x*z)
    den = 1+y+z+x*(1+z)**2
    return(np.sqrt(num/den))

# def rel_e(x,y,z):
#     '''
#     Relative error in z estimate
#     x,y,z: non-dim parameters
#     '''
#     term1 = y*z*(1 + x + x*z)/((1 + z)**2*(1 + y + z + x*(1 + z)**2))
#     term2 = (1 + y + z - y*z + x**2*(1 + z)**3 -
#   x*(1 + z)*(-2*(1 + z) + y*(-1 + 2*z)))**2/(2*(1 + z)**2*(1 + x +
#    x*z)**2*(1 + y + z + x*(1 + z)**2)**2)
#     err = 1/(term1+term2)
#     return(err)

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

#functions below are for numerical approx of Fisher information w/ normal distr
def log(m,x,y,z):
    '''
    Log-likelihood of P_normal w/ out additive constants indep. of z
    m: number or molecules
    x,y,z: non-dim parameters
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    term = np.log(var) + (m-mean)**2/var
    term *= 1/2
    return(term)

def der(m,x,y,z):
    '''
    Numerical derivative of log-liklihood
    m: number or molecules
    x,y,z: non-dim parameters
    '''
    h =1e-10 # spacing for derivative
    d = log(m,x,y,z+h) - log(m,x,y,z-h) #numerator
    return(d/(2*h)) #difference quotient

# def Fisher1(x,y,z):
#     '''
#     Calculating relative error w/ inverse-Fisher, using P_normal approx
#     x,y,z: non-dim parameters
#     '''
#     pss = z/(1+z)
#     mean  = y*pss
#     var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
#
#     m1 = mean - 3.5*np.sqrt(var) #lower bound for sum in m
#     m2 = mean + 3.5*np.sqrt(var) #upper bound
#     m1 = max(0,np.floor(m1)) #must be int >= 0
#     m2 = np.ceil(m2) #make it an int
#
#     m = np.arange(m1,m2,1) #array of m to sum in
#     terms = der(m,x,y,z)**2*P_normal(m,x,y,z) #terms in mean of derivative^2 w.r.t. z
#     summ = np.sum(terms) #find the mean
#     return(1/(summ*z**2))


'----------------------------------------------------------------'
#variables must be array or float or int for this to work
#numbers are kinda weird to avoid hitting poles of gamma function

tf = input("Manually pick contour? Input [0] for 'no', [1] for'yes':")
tf = int(tf)

data = np.load('variables.npz', allow_pickle=True)

x,y,z = data['x'], data['y'], data['z']

if x.size == 1:
    x = float(x)
    print('x =', x)
elif y.size == 1:
    y = float(y)
    print('y =', y)
elif z.size == 1:
    z = float(z)
    print('z =', z)

x1 = 'x' #will need these for if statements later
y1 = 'y'
z1 = 'z'
'----------------------------------------------------------------'
#find the arrays to plot
v0 = OrderedDict() #variable dictionary, will not be edited
v0['x'] = x
v0['y'] = y
v0['z'] = z

v = OrderedDict() #variable dictionary, will be eddited
v['x'] = x
v['y'] = y
v['z'] = z

for i in v: #get rid of the variable not on axis
    j = v[i]
    if isinstance(j, float) or isinstance(j, int):
        v.pop(i)
        break

plot_items = list(v.items()) #plotting names and values

a, b = plot_items[0][1], plot_items[1][1] #plotting variables
indep_name, dep_name = plot_items[0][0], plot_items[1][0] #names of variables to plot

a = np.array(a) #need to recast into arrays, not sure why
b = np.array(b)

'----------------------------------------------------------------'
print('Plotting '+indep_name+' vs. '+dep_name)

### Relative error in normal distr. approx. ###
A, B = np.meshgrid(a,b) #meshgrid for contour plot
#print(a,b)
#exit()
err_n = np.load('err_norm.npy', allow_pickle=True)
err_a = np.load('err_analytic.npy', allow_pickle=True)
# err_n = err_n[:-1, :-1]

#levels = MaxNLocator(nbins=5).tick_values(err.min(), 1) #setting contour levels
levels=[1e0,1e1,1e2,1e3]
fmt = LogFormatterSciNotation()
fmt.create_dummy_axis()
#levelsf = MaxNLocator(nbins=60).tick_values(err.min(), err.max()) #setting contourf levels
levelscheck = [2.0] #contours for regimes

#mesh of regimes
if indep_name == x1 and dep_name == y1:
    R1_plot = R1(A,B,z)
elif indep_name == x1 and dep_name == z1:
    R1_plot = R1(A,y,B)
elif indep_name == y1 and dep_name == z1:
    R1_plot = R1(x,A,B)

CS1 = plt.contour(A, B, R1_plot, colors='red', linestyles='dashed', levels=levelscheck, linewidths=0.8) #region of validity

plt.pcolormesh(A, B, err_n,norm=colors.LogNorm(vmin=err_n.min(), vmax=err_a.max())) #coloured plot
plt.colorbar()

CS = plt.contour(A, B, err_n, levels=levels, colors='white', linewidths=0.5) #only contours
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt, manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt)

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_norm_'+indep_name+dep_name)
plt.show()
plt.close()

### Plotting regime of validity ###
CS1 = plt.contour(A, B, R1_plot, colors='red', linestyles='dashed', levels=levelscheck)#region of validity
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = fmt, manual=True)
else:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = fmt)

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
plt.savefig('regime_'+indep_name+dep_name)
plt.show()
plt.close()

### Analytic relative error ###
#err= np.zeros([len(b),len(a)])
# err_a = err_a[:-1, :-1]

# CS1 = plt.contour(A, B, R1_plot, colors='red', linestyles='dashed', levels=levelscheck, linewidths=0.8) #region of validity

plt.pcolormesh(A, B, err_a,norm=colors.LogNorm(vmin=err_n.min(), vmax=err_a.max())) #coloured plot
plt.colorbar()

CS = plt.contour(A, B, err_a, levels=levels, colors='white', linewidths=0.5) #only contours
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt, manual=True) #if manual=True, use command line for GUI to pop up
else:
    plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = fmt)

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_analytic_'+indep_name+dep_name)#,format='eps')
plt.show()
plt.close()

### Plotting difference ###
diff = np.abs(err_a-err_n)/err_a

CS1 = plt.contour(A, B, R1_plot, colors='red', linestyles='dashed', levels=levelscheck)#region of validity
plt.pcolormesh(A, B, diff,norm=colors.LogNorm(vmin=diff.min(), vmax=diff.max()))#vmax=diff.max()) #coloured plot
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

if tf == 1:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = fmt, manual=True)
else:
    plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = fmt)

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
plt.savefig('difference_'+indep_name+dep_name)
plt.show()
plt.close()
