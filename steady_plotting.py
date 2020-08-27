#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:21:01 2020

@author: tim

Systematic plotting routine for non-steady state. 
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from laguerre_testing import Fisher, aFisher, rel_e #Fisher function
from collections import OrderedDict
import matplotlib.colors as colors

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

'----------------------------------------------------------------'
#variables must be array or float or int for this to work
#numbers are kinda weird to avoid hitting poles of gamma function
l = int(1e2+1)
x = np.logspace(-4.1,4.51,l) #x values
y = np.logspace(-2.1,2.51,l) #y values
z = 2.5  #z values

est = 'acc' #which error to calculate

x1 = 'x' #will need these for if statements later
y1 = 'y'
z1 = 'z'
'----------------------------------------------------------------'

np.savez('variables',x=x,y=y,z=z) #save for future use

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

'----------------------------------------------------------------'
print('Plotting '+indep_name+' vs. '+dep_name)
### Relative error in normal distr. approx. ###
A, B = np.meshgrid(a,b) #meshgrid for contour plot

#mesh of relative error, must check which vars are arrays
if indep_name == x1 and dep_name == y1:
    err = rel_e(A,B,z,est)
elif indep_name == x1 and dep_name == z1:
    err = rel_e(A,y,B,est)
elif indep_name == y1 and dep_name == z1:
    err = rel_e(x,A,B,est)

np.save('err_norm',err)#_'+indep_name+dep_name, err)
err = err[:-1, :-1]

#levels = MaxNLocator(nbins=5).tick_values(err.min(), 1) #setting contour levels
levels=[0.8,1.0,1.2,1.4]
levelsf = MaxNLocator(nbins=60).tick_values(err.min(), err.max()) #setting contourf levels
levelscheck = [0.1, 0.2, 0.3,0.4] #contours for regimes
max = 2 #max color for colorbar
#mesh of regimes
if indep_name == x1 and dep_name == y1:
    R1_plot = R1(A[:-1, :-1],B[:-1, :-1],z)
elif indep_name == x1 and dep_name == z1:
    R1_plot = R1(A[:-1, :-1],y,B[:-1, :-1])
elif indep_name == y1 and dep_name == z1:
    R1_plot = R1(x,A[:-1, :-1],B[:-1, :-1])

CS1 = plt.contour(A[:-1, :-1], B[:-1, :-1], R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#    plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A[:-1, :-1], B[:-1, :-1], err,norm=colors.LogNorm(vmin=err.min(), vmax=err.max())) #coloured plot
plt.colorbar()

CS = plt.contour(A[:-1, :-1], B[:-1, :-1], err, levels=levels, colors='white', linewidths=0.5) #only contours
plt.xscale('log')
plt.yscale('log')

plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True) #if manual=True, use command line for GUI to pop up

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
plt.title(est)
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_norm_'+indep_name+dep_name)
plt.show()
plt.close()

### Plotting regime of validity ###
CS1 = plt.contour(A[:-1, :-1], B[:-1, :-1], R1_plot, colors='red', linestyles='dashed', levels=levelscheck)#region of validity

plt.xscale('log')
plt.yscale('log')

plt.clabel(CS1, CS1.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True)

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
plt.savefig('regime_'+indep_name+dep_name)
# plt.show()
plt.close()
### Analytic relative error ###
err= np.zeros([len(b),len(a)])

if indep_name == x1 and dep_name == y1:
    for i in range(len(a)): #must plot manually; numpy operations do not work with Fisher function
        if i%10==0:
            print(i, 'out of', len(a), 'rows') #just to see how fast it is
        for j in range(len(b)):
            err[i,j] = aFisher(a[j],b[i],z,est)
elif indep_name == x1 and dep_name == z1:
    for i in range(len(a)):
        if i%10==0:
            print(i, 'out of', len(a), 'rows')
        for j in range(len(b)):
            err[i,j] = aFisher(a[j],y,b[i],est)
elif indep_name == y1 and dep_name == z1:
    for i in range(len(a)):
        if i%10==0:
            print(i, 'out of', len(a), 'rows')
        for j in range(len(b)):
            err[i,j] = aFisher(x,a[j],b[i],est)

np.save('err_analytic',err)#_'+indep_name+dep_name, err)
err = err[:-1, :-1]

CS1 = plt.contour(A[:-1, :-1], B[:-1, :-1], R1_plot, colors='red', linestyles='dashed', levels=levelscheck) #region of validity
#plt.clabel(CS1, CS1.levels, inline=True, fontsize=10, fmt = '%1.1f')

plt.pcolormesh(A[:-1, :-1], B[:-1, :-1], err,norm=colors.LogNorm(vmin=err.min(), vmax=err.max())) #coloured plot
plt.colorbar()

CS = plt.contour(A[:-1, :-1], B[:-1, :-1], err, levels=levels, colors='white', linewidths=0.5) #only contours
plt.xscale('log')
plt.yscale('log')

plt.clabel(CS, CS.levels, inline=True, fontsize=12, fmt = '%1.1f')#, manual=True) #if manual=True, use command line for GUI to pop up

plt.xlabel('$'+indep_name+'$')
plt.ylabel('$'+dep_name+'$')
#plt.title('Estimated relative error for $y$=%s' %y)
plt.savefig('err_analytic_'+indep_name+dep_name)#,format='eps')
# plt.show()
plt.close()
