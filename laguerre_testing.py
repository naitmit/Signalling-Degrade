#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:16:40 2020

@author: tim

For calculating probability distributions, Fisher info, CRLB and related formulae
Esp. for evaluating Laguerre solutions numerically

Using mpmath for accuracy: n! and 1F1 expressions cause floating point errors otherwise.
Refer to http://mpmath.org/doc/current/. Can  be installed with Conda.
Wherever mpmath is used, must be careful when using numpy arrays; numpy operations (array addition, etc.)
do not work. Try np.vectorize.

NOTE: for some parameter combinations, calculations may be undefined due to the special functions not existing.
This usually happens in a combination where z is an integer, giving a pole in the gamma function.
If so, make it a float very close to the original integer. It seems that the limit exists and provides the
right answer (need to look into it more).

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpmath import *
from params import DEFAULT_PARAMS
from formulae import ns_mean, ns_var
from time import time

c, kon, koff, kp, x, pss, r, kd, d_m, k_f = DEFAULT_PARAMS.unpack()

mp.dps = 50 #precision


'-----------------------Steady State---------------------------'
def P(m):
    '''
    Find the probability of finding m molecules, using the generating function
    P(m) = sum_i 1/m! (d/ds)^m G_i(s=0).
    Uses normal Python capabilities; limited usage for beyond certain parameter space
    m: number of molcules
    '''
    m1 = m+1
    P0 = 1/(1+z) #eq unoccupancy, coeff in G0
    L0 = sp.eval_genlaguerre(-z*x, z*x+x, 0) #other factor
    c_L = P0/L0 #coeff of Laguerre function
    L_m = sp.eval_genlaguerre(-z*x-m, z*x+x+m, -y) #mth derivative of Laguerre at s=0, using its derivative property + chain rule
    L_m1 = sp.eval_genlaguerre(-z*x-m1, z*x+x+m1, -y) #m+1th derivative of Laguerre at s=0
    P = (z+1+m/x)*L_m + y*L_m1/x
    P *= (-y)**m
    return(c_L*P/sp.factorial(m,exact=True))

def R1(x,y,z):
    '''
    Ratio to check for normal distr: <n>/std.dev
    x,y,z: non-dim parameters
    '''
    num = y*z*(1+x+x*z)
    den = 1+y+z+x*(1+z)**2
    return(np.sqrt(num/den))

def P_mp(m,x,y,z): #TODO find a way to bring numpy array operations if possible
    '''
    Find the probability of finding m molecules, using the generating function
    P(m) = sum_i 1/m! (d/ds)^m G_i(s=0).
    Using mpmath for precision.
    We use the def'n of L in terms of 1F1
    x,y,z: non-dim parameters
    m: number of molcules
    '''
    m1 = m+1
    P0 = 1/(1+z) #eq unoccupancy, coeff in G0

    L0_1 = binomial(x,-z*x) #coeff of Laguerre
    L0_2 = hyp1f1(z*x,z*x+x+1,0) #hypergeometric fn

    L0 = fmul(L0_1,L0_2) #L0

    #mth Laguerre
    L_m_1 = binomial(x,-z*x-m)
    L_m_2 = hyp1f1(z*x+m,z*x+x+m+1,-y)
    L_m = fmul(L_m_1, L_m_2)

    #m+1th Laguerre
    L_m1_1 = binomial(x,-z*x-m1)
    L_m1_2 = hyp1f1(z*x+m1,z*x+x+m1+1,-y)
    L_m1 = fmul(L_m1_1, L_m1_2)

    L_m = fmul(L_m, z+1+m/x) #get each term
    L_m1 = fmul(L_m1, y/x)

    P = fadd(L_m, L_m1)
    P = fmul(P, power(-y,m))

    c_L = fdiv(P0,L0)
    P = fmul(c_L,P)
    P = fdiv(P, fac(m))
    return(float(P))

def P_normal(m,x,y,z):
    '''
    Normal approximation.
    x,y,z: non-dim parameters
    m: number of molcules
    '''
    pss = z/(1+z)
    mean  = y*pss
#    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    var = y*z*(1 + y + z + x*(1 + z)**2)/((1 + z)**2*(1 + x + x*z))
    return(np.sqrt(1/(2*np.pi*var))*np.exp(-(m-mean)**2/(2*var)))

def rel_e(x,y,z, estimate='acc'):
    '''
    Relative error in z estimate using Gaussian. Probably already in another file.
    x,y,z: non-dim parameters
    '''
    if estimate == 'acc':
        term1 = y*z*(1 + x + x*z)/((1 + z)**2*(1 + y + z + x*(1 + z)**2))
        term2 = (1 + y + z - y*z + x**2*(1 + z)**3 -
      x*(1 + z)*(-2*(1 + z) + y*(-1 + 2*z)))**2/(2*(1 + z)**2*(1 + x +
       x*z)**2*(1 + y + z + x*(1 + z)**2)**2)
        err = 1/(term1+term2)
        return(err)
    else:
        assert estimate =='spec'
        num = (2*(1 + z)**2*(1 + x + x*z)**2*(1 + y + z + x*(1 + z)**2)**2)
        den1 = 2*y*(1 + z) + (1 + z)**2 + x**4*(1 + z)**5*(1 + z + 2*y*z) + y**2*(1 + z**2)
        den2 = 2*x**3*(1 + z)**3*(y**2*z + 2*(1 + z)**2 + y*(1 + z)*(2 + 3*z))
        den3 = 2*x*(1 + z)*(2*(1 + z)**2 + y*(1 + z)*(4 + z) + y**2*(2 + z**2))
        den4 = x**2*(1 + z)**2*(6*(1 + z)**2 + 2*y*(1 + z)*(5 + 3*z) + y**2*(4 + z*(2 + z)))
        den = den1+den2+den3+den4
        return(num/den)
        

def g(m,x,y,z):
    '''
    g according to P(n) = (-y)^n g(n)/n!. Used to simplify Fisher info.
    x,y,z: non-dim parameters
    m: number of molecules
    '''
    m1 = m+1
    P0 = 1/(1+z) #eq unoccupancy, coeff in G0

    L0_1 = binomial(x,-z*x) #coeff of Laguerre
    L0_2 = hyp1f1(z*x,z*x+x+1,0) #hypergeometric fn

    L0 = fmul(L0_1,L0_2) #L0

    #mth Laguerre
    L_m_1 = binomial(x,-z*x-m)
    L_m_2 = hyp1f1(z*x+m,z*x+x+m+1,-y)
    L_m = fmul(L_m_1, L_m_2)

    #m+1th Laguerre
    L_m1_1 = binomial(x,-z*x-m1)
    L_m1_2 = hyp1f1(z*x+m1,z*x+x+m1+1,-y)
    L_m1 = fmul(L_m1_1, L_m1_2)

    L_m = fmul(L_m, z+1+m/x) #get each term
    L_m1 = fmul(L_m1, y/x)

    P = fadd(L_m, L_m1)
    c_L = fdiv(P0,L0)
    P = fmul(c_L,P)
    return(fabs(P))

def g_c(m,x,y,z):
    '''
    Find the derivative of g w.r.t. z (equiv to c in the end, see write-up).
    x,y,z: non-dim parameters
    m: number of molecules
    '''
    der = diff(g, (m,x,y,z),(0,0,0,1),addprec=20)
    return(der)

def Fisher(x,y,z):
    '''
    Analytic inverse-Fisher based on truncated sum approx.
    x,y,z: non-dim parameters
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    acc = 5 #number of std dev for accuracy
    m1 = mean - acc*np.sqrt(var) #lower bound for sum in m
    m2 = mean + acc*np.sqrt(var) #upper bound
    m1 = max(0,np.floor(m1)) #must be int >= 0
    m2 = np.ceil(m2) #make it an int

    m = np.arange(m1,m2,1) #array of m to sum in

    l = len(m) #iterate in this
    summ = 0 #initiate sum
    for i in range(l): #iterate in m array
        term = power(g_c(m[i], x, y, z), 2) #square the derivative
        term = fdiv(term, g(m[i],x,y,z)) #divide by g
        term = fmul(term, power(y,m[i]))
        term = fdiv(term, fac(m[i]))
#        if i == l-1:
#            print('Last term', term)
        summ = fadd(summ, term)
    summ = fdiv(1,summ)
    summ = fdiv(summ,z**2)
    return(float(summ))


def aFisher(x,y,z,estimate='acc'):
    '''
    Analytic scaled error from inverse-Fisher based on truncated sum, with more numerical precision
    using adaptive summing
    (neglecting prescision lost from floating-point operations - need to set mpmath accuracy
    for that)
    (x,y,z): parameters
    estimate: 'acc' for accuracy (c), 'spec' for specificity (k_off)
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    sig = np.sqrt(var)
    acc = 5 #initial number of std dev for range of sum
    m1 = mean - acc*sig #lower bound for sum in m
    m2 = mean + acc*sig #upper bound
    m1 = max(0,np.floor(m1)) #must be int >= 0
    m2 = np.ceil(m2) #make it an int

    m = np.arange(m1,m2,1) #array of m to sum in

    l = len(m) #iterate in this
    summ = 0 #initiate sum

    df = 1e-6 #desired numerical precision (NOT estimation accuracy)
    rel_a = 1 #intiate numerical precision
    dm = 1e-2 #increment to increase sum in both directions, for adding tails of dm*sig to sum
    k = 1 #for the loop
    if estimate == 'acc': #accuracy calculations
        while rel_a > df:
            if k == 1: #first iteration
                for i in range(l): #iterate in m array
                    term = power(g_c(m[i], x, y, z), 2) #square the derivative
                    term = fdiv(term, g(m[i],x,y,z)) #divide by g
                    term = fmul(term, power(y,m[i]))
                    term = fdiv(term, fac(m[i]))
            #        if i == l-1:
            #            print('Last term', term)
                    summ = fadd(summ, term)
                rel_a = float(fdiv(term,summ))
                k = 2 #stop from going through this loop again
            else: #if accuracy is still not as desired
                if m1 == 0: #cannot include terms for m1<0, only include terms larger than m2
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = m_ftail[i] #value of m
                        term = power(g_c(mi, x, y, z), 2) #square the derivative
                        term = fdiv(term, g(mi,x,y,z)) #divide by g
                        term = fmul(term, power(y,mi))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    rel_a = float(fdiv(term,summ))
                else:
                    #first sum the front tail
                    m0 = max(0, np.floor(m1 - sig*dm)) #new end of sum
                    m_btail = np.arange(m0,m1) #back tail of sum
                    bl = len(m_btail) #length of back tail to iterate in
                    for i in range(bl): #first sum the back tail
                        j = bl-i-1 #sum in decreasing m
                        mj = m_btail[j]
                        term = power(g_c(mj, x, y, z), 2) #square the derivative
                        term = fdiv(term, g(mj,x,y,z)) #divide by g
                        term = fmul(term, power(y,mj))
                        term = fdiv(term, fac(mj))
                        summ = fadd(summ, term)
                    m1 = m0 #assign new back of sum
                    termb = term #record very back term
    
                    #now we sum the front tail
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = m_ftail[i] #value of m
                        term = power(g_c(mi, x, y, z), 2) #square the derivative
                        term = fdiv(term, g(mi,x,y,z)) #divide by g
                        term = fmul(term, power(y,mi))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    termf = term #record very front term
    
                    ac1 = float(fdiv(termb,summ)) #accuracy in back term
                    ac2 = float(fdiv(termf,summ)) #accuracy in front term
                    rel_a =max(ac1,ac1) #use largest
            # print(rel_a) #just to check
        summ = fdiv(1,summ)
        summ = fdiv(summ,z**2)
        return(float(summ))
    
    else: #specificity calculations
        assert estimate == 'spec'
        while rel_a > df:
            if k == 1: #first iteration
                for i in range(l): #iterate in m array
                    derz = diff(g, (m[i],x,y,z),(0,0,0,1),addprec=20) #z derivative
                    derx = diff(g, (m[i],x,y,z),(0,1,0,0),addprec=20) #x derivative
                    term1 = fmul(x,derx)
                    term2 = fmul(z,derz)
                    term = fsub(term1,term2)
                    term = power(term, 2) #square the derivative
                    term = fdiv(term, g(m[i],x,y,z)) #divide by g
                    term = fmul(term, power(y,m[i]))
                    term = fdiv(term, fac(m[i]))
            #        if i == l-1:
            #            print('Last term', term)
                    summ = fadd(summ, term)
                rel_a = float(fdiv(term,summ))
                k = 2 #stop from going through this loop again
            else: #if accuracy is still not as desired
                if m1 == 0: #cannot include terms for m1<0, only include terms larger than m2
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = m_ftail[i] #value of m
                        derz = diff(g, (mi,x,y,z),(0,0,0,1),addprec=20) #z derivative
                        derx = diff(g, (mi,x,y,z),(0,1,0,0),addprec=20) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term, 2) #square the derivative
                        term = fdiv(term, g(mi,x,y,z)) #divide by g
                        term = fmul(term, power(y,mi))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    rel_a = float(fdiv(term,summ))
                else:
                    #first sum the back tail
                    m0 = max(0, np.floor(m1 - sig*dm)) #new end of sum
                    m_btail = np.arange(m0,m1) #back tail of sum
                    bl = len(m_btail) #length of back tail to iterate in
                    for i in range(bl): #first sum the back tail
                        j = bl-i-1 #sum in decreasing m
                        mj = m_btail[j]
                        derz = diff(g, (mj,x,y,z),(0,0,0,1),addprec=20) #z derivative
                        derx = diff(g, (mj,x,y,z),(0,1,0,0),addprec=20) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term, 2) #square the derivative
                        term = fdiv(term, g(mj,x,y,z)) #divide by g
                        term = fmul(term, power(y,mj))
                        term = fdiv(term, fac(mj))
                        summ = fadd(summ, term)
                    m1 = m0 #assign new back of sum
                    termb = term #record very back term
    
                    #now we sum the front tail
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = m_ftail[i] #value of m
                        derz = diff(g, (mi,x,y,z),(0,0,0,1),addprec=20) #z derivative
                        derx = diff(g, (mi,x,y,z),(0,1,0,0),addprec=20) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term, 2) #square the derivative
                        term = fdiv(term, g(mi,x,y,z)) #divide by g
                        term = fmul(term, power(y,mi))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    termf = term #record very front term
    
                    ac1 = float(fdiv(termb,summ)) #accuracy in back term
                    ac2 = float(fdiv(termf,summ)) #accuracy in front term
                    rel_a =max(ac1,ac1) #use largest
            # print(rel_a) #just to check
        summ = fdiv(1,summ)
        # summ = fdiv(summ,z**2)
        return(float(summ))
    
def Fisher2(A,B,C):
    '''
    Analytic inverse-Fisher based on sum approx using new dimless parameters
    '''
    x, y, z = B*C, C, A/B
    return(Fisher(x,y,z))

def aFisher2(A,B,C):
    '''
    Analytic inverse-Fisher based on sum approx using new dimless parameters
    '''
    x, y, z = B*C, C, A/B
    return(aFisher(x,y,z))

def rel_e2(A,B,C):
    '''
    Relative error in z estimate using new dimless parameters
    x,y,z: non-dim parameters
    '''
    x, y, z = B*C, C, A/B
    return(rel_e(x,y,z))



'--------------------------Dynamical-----------------------------'
def g_ns(t, s, params=DEFAULT_PARAMS):
    '''
    Non-steady state generating function G(t,s).
    t: time
    s: generating function variable
    '''
    c, kon, koff, kp, x, pss, r, kd, d_m, k_f = DEFAULT_PARAMS.unpack()
    ''
    exp_t  =exp(-kd*t) #exponential factor
    ckon = c*kon
    arg = fmul(exp_t, kp*(s-1)/kd)

    ''
    num1_1 = hyp1f1(-koff/kd, (kd - koff - ckon)/kd, arg)
    num1_1 = fmul(num1_1, (kd - koff - ckon))
    num1_1 = fmul(num1_1, exp(kd*t))

    num1_2_1 = hyp1f1(ckon/kd, (kd + koff + ckon)/kd, kp*(-1 + s)/kd)
    num1_2_1 = fmul(num1_2_1, kd + koff + ckon)
    num1_2_1 = fmul(num1_2_1, koff + ckon)
    num1_2_2 = hyp1f1(1 + (ckon)/kd, (2*kd + koff + ckon)/kd, (kp*(-1 + s))/kd)
    num1_2_2 = fmul(num1_2_2, ckon*kp*(-1 + s))
    num1_2 = fadd(num1_2_1, num1_2_2)

    num1 = fmul(num1_1, num1_2)
    ''
    num2_1 = hyp1f1(1 - koff/kd, -((-2*kd + koff + ckon)/kd), arg)
    num2_1 = fmul(num2_1, kp*(s-1))

    num2_2_1 = hyp1f1((ckon)/kd, (kd + koff + ckon)/kd, (kp*(-1 + s))/kd)
    num2_2_1 = fmul((koff + ckon)*(kd + koff + ckon), num2_2_1)
    num2_2_2 = hyp1f1(1 + (ckon)/kd, (2*kd + koff + ckon)/kd, (kp*(-1 + s))/kd)
    num2_2_2 = fmul(num2_2_2, c*kon*kp*(-1 + s))
    num2_2 = fadd(num2_2_1, num2_2_2)

    num2 = fmul(num2_1,num2_2)
    ''
    num3_1 = hyp1f1(1 - koff/kd, -((-2*kd + koff + ckon)/kd), (kp*(-1 + s))/kd)
    num3_1 = fmul(num3_1, -c*kon*kp*(-1 + s))

    num3_2_1 = hyp1f1((ckon)/kd, (kd + koff + ckon)/kd, arg)
    num3_2_1 = fmul(num3_2_1, exp(t*(-koff-ckon+kd)))
    num3_2_1 = fmul(num3_2_1, (kd + koff + ckon))
    num3_2_2 = hyp1f1(1 + (ckon)/kd, (2*kd + koff + ckon)/kd, arg)
    num3_2_2 = fmul(exp(-t*(koff + ckon)),num3_2_2)
    num3_2_2 = fmul(num3_2_2, kp*(-1 + s))
    num3_2 = fadd(num3_2_1, num3_2_2)

    num3 = fmul(num3_2, num3_1)
    ''
    num = fadd(num1,num2)
    num = fadd(num,num3)
    ''
    den1_1 = hyp1f1(1 - koff/kd, -((-2*kd + koff + ckon)/kd), arg)
    den1_2 = hyp1f1((ckon)/kd, (kd + koff + ckon)/kd, arg)
    den1_3 = koff*(kd + koff + ckon)*kp*(-1 + s)
    den1 = fmul(den1_1,den1_2)
    den1 = fmul(den1, den1_3)

    den2_1 = hyp1f1(-(koff/kd), (kd - koff - ckon)/kd, arg)
    den2_1 = fmul(den2_1, kd - koff - ckon)

    den2_2_1 = hyp1f1((ckon)/kd, (kd + koff + ckon)/kd, arg)
    den2_2_1 = fmul(den2_2_1, (koff + ckon)*(kd + koff + ckon))
    den2_2_1 = fmul(den2_2_1, exp(kd*t))

    den2_2_2 = hyp1f1(1 + (ckon)/kd, (2*kd + koff + ckon)/kd,arg)
    den2_2_2 = fmul(den2_2_2,ckon*kp*(-1 + s))

    den2_2 = fadd(den2_2_1, den2_2_2)

    den2 = fmul(den2_1,den2_2)
    ''
    den = fadd(den1,den2)

    G = fdiv(num,den)
    return(G)

def g_ns2(T, s, x, y, z):
    '''
    Non-steady state generating function using non-dim parameters.
    T: kd*t
    s: generating function variable
    x: koff/kd
    y: kp/kd
    z: kon*c/off
    '''
    ''
    exp_t  =exp(-T) #exponential factor
    arg = fmul(exp_t, y*(s-1))

    ''
    num1_1 = hyp1f1(-x, (1 - x - z*x), arg)
    num1_1 = fmul(num1_1, (1 - x - z*x))
    num1_1 = fmul(num1_1, exp(T))

    num1_2_1 = hyp1f1(z*x, (1 + x + z*x), y*(-1 + s))
    num1_2_1 = fmul(num1_2_1, 1 + x + z*x)
    num1_2_1 = fmul(num1_2_1, x + z*x)
    num1_2_2 = hyp1f1(1 + z*x, (2 + x + z*x), (y*(-1 + s)))
    num1_2_2 = fmul(num1_2_2, z*x*y*(-1 + s))
    num1_2 = fadd(num1_2_1, num1_2_2)

    num1 = fmul(num1_1, num1_2)
    ''
    num2_1 = hyp1f1(1 - x, -((-2 + x + z*x)), arg)
    num2_1 = fmul(num2_1, y*(s-1))

    num2_2_1 = hyp1f1(z*x, (1 + x + z*x), (y*(-1 + s)))
    num2_2_1 = fmul((x + z*x)*(1 + x + z*x), num2_2_1)
    num2_2_2 = hyp1f1(1 + z*x, (2 + x + z*x), (y*(-1 + s)))
    num2_2_2 = fmul(num2_2_2, z*x*y*(-1 + s))
    num2_2 = fadd(num2_2_1, num2_2_2)

    num2 = fmul(num2_1,num2_2)
    ''
    num3_1 = hyp1f1(1 - x, -((-2 + x + z*x)), (y*(-1 + s)))
    num3_1 = fmul(num3_1, -z*x*y*(-1 + s))

    num3_2_1 = hyp1f1(z*x, (1 + x + z*x), arg)
    num3_2_1 = fmul(num3_2_1, exp(T*(-x-z*x+1)))
    num3_2_1 = fmul(num3_2_1, (1 + x + z*x))
    num3_2_2 = hyp1f1(1 + z*x, (2 + x + z*x), arg)
    num3_2_2 = fmul(exp(-T*(x + z*x)),num3_2_2)
    num3_2_2 = fmul(num3_2_2, y*(-1 + s))
    num3_2 = fadd(num3_2_1, num3_2_2)

    num3 = fmul(num3_2, num3_1)
    ''
    num = fadd(num1,num2)
    num = fadd(num,num3)
    ''
    den1_1 = hyp1f1(1 - x, -((-2 + x + z*x)), arg)
    den1_2 = hyp1f1(z*x, (1 + x + z*x), arg)
    den1_3 = x*(1 + x + z*x)*y*(-1 + s)
    den1 = fmul(den1_1,den1_2)
    den1 = fmul(den1, den1_3)

    den2_1 = hyp1f1(-(x), (1 - x - z*x), arg)
    den2_1 = fmul(den2_1, 1 - x - z*x)

    den2_2_1 = hyp1f1(z*x, (1 + x + z*x), arg)
    den2_2_1 = fmul(den2_2_1, (x + z*x)*(1 + x + z*x))
    den2_2_1 = fmul(den2_2_1, exp(T))

    den2_2_2 = hyp1f1(1 + z*x, (2 + x + z*x),arg)
    den2_2_2 = fmul(den2_2_2,z*x*y*(-1 + s))

    den2_2 = fadd(den2_2_1, den2_2_2)

    den2 = fmul(den2_1,den2_2)
    ''
    den = fadd(den1,den2)

    G = fdiv(num,den)
    return(G)


def P_ns(n, t, F=False):
    '''
    Dynamical probability distr
    n: no. of molecules
    t: time
    F: bool for float or not
    '''
    num = diff(g_ns, (t, 0), (0,n))
    den = fac(n)
    prob = fdiv(num,den)
    if F is True:
        prob = float(prob)
    return(prob)

def P_ns2(n, T, x, y, z, F=False):
    '''
    Dynamical probability distr using non-dim parameters.
    n: no. of molecules
    T: kd*t
    '''
    num = diff(g_ns2, (T, 0, x, y, z), (0, n, 0, 0, 0))
    den = fac(n)
    prob = fdiv(num,den)
    if F is True:
        prob = float(prob)
    return(prob)

def P_ns3(n, T, x, y, z, F=False):
    '''
    For Fisher info, derivative of probability w.r.t. z using non-dim parameters.
    n: no. of molecules
    T: kd*t
    '''
    num = diff(g_ns2, (T, 0, x, y, z), (0, n, 0, 0, 1))
    den = fac(n)
    prob = fdiv(num,den)
    if F is True:
        prob = float(prob)
    return(prob)

def m_ns(t):
    '''
    Dynamical mean using g_ns w/ numerical differentiation
    '''
    return(float(diff(g_ns, (t, 1), (0,1))))

def v_ns(t):
    '''
    Dynamical variance using g_ns w/ numerical differentiation
    '''
    term1 = diff(g_ns, (t, 1), (0,2))
    term2 = diff(g_ns, (t, 1), (0,1))
    term3 = fmul(term2,term2)
    v = fadd(term1,term2)
    v = fsub(v,term3)
    return(float(v))

def m_ns2(T, x, y, z):
    '''
    Dynamical mean using g_ns2 w/ numerical differentiation, using non-dim parameters
    '''
    mean = diff(g_ns2, (T, 1, x, y, z), (0, 1, 0, 0, 0))
    return(mean)

def v_ns2(T, x, y, z):
    '''
    Dynamical var using g_ns2 w/ numerical differentiation, using non-dim parameters
    '''
    term1 = diff(g_ns2, (T, 1, x, y, z), (0, 2, 0, 0, 0))
    term2 = diff(g_ns2, (T, 1, x, y, z), (0, 1, 0, 0, 0))
    term3 = fmul(term2,term2)
    v = fadd(term1,term2)
    v = fsub(v,term3)
    return(v)

def ns_mean2(T, x, y, z):
    '''
    Calculates dynamical mean using exact expression. Used for testing in traj sim code.
    '''
    fac1 = 1 - np.exp(-(x + x*z)*T) + (np.exp(-T) - 1)*(x + x*z)
    fac2 = -x*z*y
    den = (x + x*z)*(-1 + x + x*z)
    return(fac1*fac2/den)

def ns_var2(T, x, y, z):
    '''
    Calculates dynamical variance using exact expression. Used for testing in traj sim code.
    '''
    fac1 = x*z*y

    # a = np.exp(2*T)
    a1 = -c*kon*(-2 + x + x*z)*(1 + x + x*z)*y*np.exp(-2*(x + x*z)*T)
    a2_1 = -(2 - x - x*z)*(-1 + x + x*z)**2
    a2_2 = (x + x*z)*(1 + x + x*z) + x*y
    a2 = a2_1*a2_2
    a3_1 = np.exp(-(x + x*z)*T)*(1 - x - x*z)*(1 + x + x*z)
    a3_2 = (2 - x - x*z)*(x + x*z) + 2*(1 - x - x*z)*(x - x*z)*y
    a3 = a3_1*a3_2
    term1 = (a1+a2+a3)

    b = -((x + x*z)**2)
    b1 = -np.exp(-2*T)*(1 - x)*(1 + x + x*z)*y
    b2_1 = (2 - x - x*z)
    b2_2 = np.exp(-T)*(1 - x - x*z)*(1 + x + x*z) + np.exp((-1 - x - x*z)*T)*2*(1 - x + x*z)*y
    b2 = b2_1*b2_2
    term2 = b*(b1+b2)

    num = fac1*(term1+term2)

    den = (x + x*z)**2*(-2 + x + x*z)*(-1 + x +
   x*z)**2*(1 + x + x*z)
    return(num/den)

def normal_ns(n, T, x, y ,z):
    '''
    Gaussian approx of dynamical prob distr.
    '''
    var = float(v_ns2(T, x, y, z))
    mean = float(m_ns2(T, x, y, z))
    sig = np.sqrt(var)
    fac = 1/(sig*np.sqrt(2*np.pi))
    ex = np.exp(-(n - mean)**2/(2*var))
    return(fac*ex)

def Fisher_ns(T,x,y,z, estimate='acc'):
    '''
    Analytic scaled error from inverse-Fisher based on truncated sum, with more numerical precision
    using adaptive summing
    (neglecting prescision lost from floating-point operations - need to set mpmath accuracy
    for that)
    (T,x,y,z): paramters
    estimate: 'acc' for accuracy, 'spec' for specificity
    '''
    mean  = float(m_ns2(T,x,y,z))
    var = float(v_ns2(T,x,y,z))
    sig = np.sqrt(var)
    acc = 5 #initial number of std dev for range of sum
    m1 = mean - acc*sig #lower bound for sum in m
    m2 = mean + acc*sig #upper bound
    m1 = max(0,np.floor(m1)) #must be int >= 0
    m2 = np.ceil(m2) #make it an int

    m = np.arange(m1,m2,1, int) #array of m to sum in
    l = len(m) #iterate in this
    summ = 0 #initiate sum

    df = 1e-4 #desired numerical precision (NOT estimation accuracy)
    rel_a = 1 #intiate numerical precision
    dm = 1e-2 #increment to increase sum in both directions, for adding tails of dm*sig to sum
    k = 1 #for the loop
    if estimate == 'acc':
        while rel_a > df:
            # print(rel_a)
            if k == 1: #first iteration
                for i in range(l): #iterate in m array
                    term = P_ns3(m[i], T, x, y, z)
                    term = fdiv(power(term, 2), P_ns2(m[i], T, x, y, z))
            #        if i == l-1:
            #            print('Last term', term)
                    summ = fadd(summ, term)
                rel_a = float(fdiv(term,summ))
                k = 2 #stop from going through this loop again
            else: #if accuracy is still not as desired
                if m1 == 0: #cannot include terms for m1<0, only include terms larger than m2
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = int(m_ftail[i]) #value of m
                        term = P_ns3(mi, T, x, y, z)
                        term = fdiv(power(term,2), P_ns2(mi, T, x, y, z))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    rel_a = float(fdiv(term,summ))
                else:
                    #first sum the front tail
                    m0 = max(0, np.floor(m1 - sig*dm)) #new end of sum
                    m_btail = np.arange(m0,m1) #back tail of sum
                    bl = len(m_btail) #length of back tail to iterate in
                    for i in range(bl): #first sum the back tail
                        j = bl-i-1 #sum in decreasing m
                        mj = int(m_btail[j])
                        term = P_ns3(mj, T, x, y, z)
                        term = fdiv(power(term,2), P_ns2(mj, T, x, y, z))
                        summ = fadd(summ, term)
                    m1 = m0 #assign new back of sum
                    termb = term #record very back term
    
                    #now we sum the front tail
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = int(m_ftail[i]) #value of m
                        term = P_ns3(mi, T, x, y, z)
                        term = fdiv(power(term,2), P_ns2(mi, T, x, y, z))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    termf = term #record very front term
    
                    ac1 = float(fdiv(termb,summ)) #accuracy in back term
                    ac2 = float(fdiv(termf,summ)) #accuracy in front term
                    rel_a =max(ac1,ac1) #use largest
            # print(rel_a) #just to check
        summ = fdiv(1,summ)
        summ = fdiv(summ,z**2)
        return(float(summ))
    else: #specificity calculations
        assert estimate == 'spec'
        while rel_a > df:
            print(rel_a)
            if k == 1: #first iteration
                for i in range(l): #iterate in m array
                    derz = diff(g_ns2, (T, 0, x, y, z), (0, m[i], 0, 0, 1)) #z derivative TODO: make this block a function
                    derx = diff(g_ns2, (T, 0, x, y, z), (0, m[i], 1, 0, 0)) #x derivative
                    term1 = fmul(x,derx)
                    term2 = fmul(z,derz)
                    term = fsub(term1,term2)
                    term = power(term,2)
                    term = fdiv(term, P_ns2(m[i],T,x,y,z))
                    term = fdiv(term, fac(m[i]))
                    summ = fadd(summ, term)
                rel_a = float(fdiv(term,summ))
                k = 2 #stop from going through this loop again
            else: #if accuracy is still not as desired
                if m1 == 0: #cannot include terms for m1<0, only include terms larger than m2
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = int(m_ftail[i]) #value of m
                        derz = diff(g_ns2, (T, 0, x, y, z), (0, mi, 0, 0, 1)) #z derivative
                        derx = diff(g_ns2, (T, 0, x, y, z), (0, mi, 1, 0, 0)) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term,2)
                        term = fdiv(term, P_ns2(mi,T,x,y,z))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    rel_a = float(fdiv(term,summ))
                else:
                    #first sum the back tail
                    m0 = max(0, np.floor(m1 - sig*dm)) #new end of sum
                    m_btail = np.arange(m0,m1) #back tail of sum
                    bl = len(m_btail) #length of back tail to iterate in
                    for i in range(bl): #first sum the back tail
                        j = bl-i-1 #sum in decreasing m
                        mj = int(m_btail[j])
                        derz = diff(g_ns2, (T, 0, x, y, z), (0, mj, 0, 0, 1)) #z derivative
                        derx = diff(g_ns2, (T, 0, x, y, z), (0, mj, 1, 0, 0)) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term,2)
                        term = fdiv(term, P_ns2(mj,T,x,y,z))
                        term = fdiv(term, fac(mj))
                        summ = fadd(summ, term)
                    m1 = m0 #assign new back of sum
                    termb = term #record very back term
    
                    #now we sum the front tail
                    m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                    m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                    fl = len(m_ftail)
                    for i in range(fl): #iterate in m array
                        mi = int(m_ftail[i]) #value of m
                        derz = diff(g_ns2, (T, 0, x, y, z), (0, mi, 0, 0, 1)) #z derivative
                        derx = diff(g_ns2, (T, 0, x, y, z), (0, mi, 1, 0, 0)) #x derivative
                        term1 = fmul(x,derx)
                        term2 = fmul(z,derz)
                        term = fsub(term1,term2)
                        term = power(term,2)
                        term = fdiv(term, P_ns2(mi,T,x,y,z))
                        term = fdiv(term, fac(mi))
                        summ = fadd(summ, term)
                    m2 = m3 #assign new end of sum
                    termf = term #record very front term
    
                    ac1 = float(fdiv(termb,summ)) #accuracy in back term
                    ac2 = float(fdiv(termf,summ)) #accuracy in front term
                    rel_a =max(ac1,ac1) #use largest
            # print(rel_a) #just to check
        summ = fdiv(1,summ)
        # summ = fdiv(summ,z**2)
        return(float(summ))
    
def Fisher_nsg_num(T, x, y, z, estimate='acc'):
    '''
    Cramer-Rao using Gaussian, using numerical mean+var
    (T,x,y,z): paramters
    estimate: 'acc' for accuracy, 'spec' for specificity
    '''
    if estimate == 'acc':
        term1 = float(diff(m_ns2, (T,x,y,z), (0,0,0,1)))
        term1 = term1**2
        term1 = term1/v_ns2(T,x,y,z)
    
        term2 = float(diff(v_ns2, (T,x,y,z), (0,0,0,1)))
        term2 = term2/v_ns2(T,x,y,z)
        term2 = term2**2
        term2 = term2/2
    
        den = term1 + term2
        return(1/(z**2*den))
    
    else:
        assert estimate == 'spec'
        term1 = x*float(diff(m_ns2, (T,x,y,z), (0,1,0,0)))-z*float(diff(m_ns2, (T,x,y,z), (0,0,0,1)))
        term1 = term1**2
        term1 = term1/v_ns2(T,x,y,z)
    
        term2 = x*float(diff(v_ns2, (T,x,y,z), (0,1,0,0)))-z*float(diff(v_ns2, (T,x,y,z), (0,0,0,1)))
        term2 = term2/v_ns2(T,x,y,z)
        term2 = term2**2
        term2 = term2/2
    
        den = term1 + term2
        return(1/den)

def Fisher_nsg_ex(T, x, y, z):
    '''
    Cramer-Rao using Gaussian, using exact mean+variance
    NOTE: seems to exhibit weird spikes, probably due to floating point errors.
    Don't think this should be used.
    '''
    dz = 1e-4
    term1 = (ns_mean2(T,x,y,z+dz) - ns_mean2(T,x,y,z))/(dz)
    term1 = term1**2
    term1 = term1/ns_var2(T,x,y,z)

    term2 = (ns_var2(T,x,y,z+dz) - ns_var2(T,x,y,z))/(dz)
    term2 = term2/ns_var2(T,x,y,z)
    term2 = term2**2
    term2 = term2/2

    den = term1 + term2
    return(1/(z**2*den))

'---------------------------Misc. Testing------------------------------------'
def aInfo(x,y,z):
    '''
    Analytic self-info based on sum approx, arbitrary numerical precision
    (neglecting prescision lost from floating-point operations - need to set mpmath accuracy
    for that)
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    sig = np.sqrt(var)
    acc = 5 #initial number of std dev for range of sum
    m1 = mean - acc*sig #lower bound for sum in m
    m2 = mean + acc*sig #upper bound
    m1 = max(0,np.floor(m1)) #must be int >= 0
    m2 = np.ceil(m2) #make it an int

    m = np.arange(m1,m2,1) #array of m to sum in

    l = len(m) #iterate in this
    summ = 0 #initiate sum

    df = 1e-6 #desired numerical precision (NOT estimation accuracy)
    rel_a = 1 #intiate numerical precision
    dm = 1e-2 #increment to increase sum in both directions
    k = 1 #for the loop
    while rel_a > df:
        if k == 1: #first iteration
            for i in range(l): #iterate in m array
                fac = P_mp(m[i],x,y,z) #square the derivative
                term = log(fac,2) #divide by g
                term = fmul(term, fac)
        #        if i == l-1:
        #            print('Last term', term)
                summ = fadd(summ, term)
            rel_a = float(fdiv(term,summ))
            k = 2 #stop from going through this loop again
        else: #if accuracy is still not as desired
            if m1 == 0: #cannot include terms for m1<0, only include terms larger than m2
                m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                fl = len(m_ftail)
                for i in range(fl): #iterate in m array
                    mi = m_ftail[i] #value of m
                    fac = P_mp(mi,x,y,z) #probability
                    term = log(fac,2) #log2 of prob
                    term = fmul(term, fac)
                    summ = fadd(summ, term)
                m2 = m3 #assign new end of sum
                rel_a = float(fdiv(term,summ))
            else:
                #first sum the front tail
                m0 = max(0, np.floor(m1 - sig*dm)) #new end of sum
                m_btail = np.arange(m0,m1) #back tail of sum
                bl = len(m_btail) #length of back tail to iterate in
                for i in range(bl): #first sum the back tail
                    j = bl-i-1 #sum in decreasing m
                    mj = m_btail[j]
                    fac = P_mp(mj,x,y,z) #probability
                    term = log(fac,2) #log2
                    term = fmul(term, fac)
                    summ = fadd(summ, term)
                m1 = m0 #assign new back of sum
                termb = term #record very back term

                #now we sum the front tail
                m3 = m2 + np.ceil(dm*sig) #expanded sum ends here
                m_ftail = np.arange(m2,m3) #m's of front tail to sum in
                fl = len(m_ftail)
                for i in range(fl): #iterate in m array
                    mi = m_ftail[i] #value of m
                    fac = P_mp(mi,x,y,z) #probability
                    term = log(fac,2) #log2
                    term = fmul(term, fac)
                    summ = fadd(summ, term)
                m2 = m3 #assign new end of sum
                termf = term #record very front term

                ac1 = float(fdiv(termb,summ)) #accuracy in back term
                ac2 = float(fdiv(termf,summ)) #accuracy in front term
                rel_a =max(ac1,ac1) #use largest
        # print(rel_a) #just to check
    return(-float(summ))

def nInfo(x,y,z):
    '''
    Self-information with Gaussian approx
    '''
    pss = z/(1+z)
    mean  = y*pss
    var = y**2*z*(x*z+1)/((1+z)*(1+x+x*z)) + mean - mean**2
    term1 = 1/2*np.log2(np.e*var*2*np.pi)
    return(term1)

def nInfo2(A,B,C):
    '''
    Analytic inverse-Fisher based on sum approx using new dimless parameters
    '''
    x, y, z = B*C, C, A/B
    return(nInfo(x,y,z))

def aInfo2(A,B,C):
    '''
    Analytic inverse-Fisher based on sum approx using new dimless parameters
    '''
    x, y, z = B*C, C, A/B
    return(aInfo(x,y,z))

def mean(t):
    der = diff(g_ns, (t,1),(0,1),addprec=20)
    return(float(der))

def realmean(t):
    mean = kp/kd * pss*(1-np.exp(-kd*t))
    return(mean)

'----------------Testing each term in P-----------------------'
def P_mp1(m,x,y,z): #TODO find a way to bring numpy array operations if possible
    '''
    Find the probability of finding m molecules, using the generating function
    P(m) = sum_i 1/m! (d/ds)^m G_i(s=0).
    Using mpmath for precision.
    We use the def'n of L in terms of 1F1
    x,y,z: non-dim parameters
    m: number of molcules
    '''
    # m1 = m+1
    P0 = 1/(1+z) #eq unoccupancy, coeff in G0

    L0_1 = binomial(x,-z*x) #coeff of Laguerre
    L0_2 = hyp1f1(z*x,z*x+x+1,0) #hypergeometric fn

    L0 = fmul(L0_1,L0_2) #L0

    #mth Laguerre
    L_m_1 = binomial(x,-z*x-m)
    L_m_2 = hyp1f1(z*x+m,z*x+x+m+1,-y)
    L_m = fmul(L_m_1, L_m_2)

    # #m+1th Laguerre
    # L_m1_1 = binomial(x,-z*x-m1)
    # L_m1_2 = hyp1f1(z*x+m1,z*x+x+m1+1,-y)
    # L_m1 = fmul(L_m1_1, L_m1_2)

    L_m = fmul(L_m, z+1+m/x) #get each term
    # L_m1 = fmul(L_m1, y/x)

    # P = fadd(L_m, L_m1)
    P = fmul(L_m, power(-y,m))

    c_L = fdiv(P0,L0)
    P = fmul(c_L,P)
    P = fdiv(P, fac(m))
    return(float(P))

def P_mp2(m,x,y,z): #TODO find a way to bring numpy array operations if possible
    '''
    Find the probability of finding m molecules, using the generating function
    P(m) = sum_i 1/m! (d/ds)^m G_i(s=0).
    Using mpmath for precision.
    We use the def'n of L in terms of 1F1
    x,y,z: non-dim parameters
    m: number of molcules
    '''
    m1 = m+1
    P0 = 1/(1+z) #eq unoccupancy, coeff in G0

    L0_1 = binomial(x,-z*x) #coeff of Laguerre
    L0_2 = hyp1f1(z*x,z*x+x+1,0) #hypergeometric fn

    L0 = fmul(L0_1,L0_2) #L0

    # #mth Laguerre
    # L_m_1 = binomial(x,-z*x-m)
    # L_m_2 = hyp1f1(z*x+m,z*x+x+m+1,-y)
    # L_m = fmul(L_m_1, L_m_2)

    #m+1th Laguerre
    L_m1_1 = binomial(x,-z*x-m1)
    L_m1_2 = hyp1f1(z*x+m1,z*x+x+m1+1,-y)
    L_m1 = fmul(L_m1_1, L_m1_2)

    # L_m = fmul(L_m, z+1+m/x) #get each term
    L_m1 = fmul(L_m1, y/x)

    # P = fadd(L_m, L_m1)
    P = fmul(L_m1, power(-y,m))

    c_L = fdiv(P0,L0)
    P = fmul(c_L,P)
    P = fdiv(P, fac(m))
    return(float(P))

if __name__ == '__main__':
    vP = np.vectorize(P_ns2)
    vPss = np.vectorize(P_mp)
    vP1 = np.vectorize(P_mp1)
    vP2 = np.vectorize(P_mp2)
    vm = np.vectorize(v_ns2)

    # t = 1
    # x1, y1, z1 = .1, 20.1, 3.5
    # x2, y2, z2 = 1.1, 20.1, 1.0
    # x3, y3, z3 = 5.1, 20.1, 5.0
    # n = np.arange(0,2*y1,1)
    
    # fig, axs = plt.subplots(3, sharex=True)
    # axs[0].plot(n, vPss(n,x1,y1,z1), label='Analytic')
    # axs[0].plot(n, P_normal(n,x1,y1,z1), label='Normal')
    # axs[0].legend()
    # axs[1].plot(n, vPss(n,x2,y2,z2), label='Analytic')
    # axs[1].plot(n, P_normal(n,x2,y2,z2), label='Normal')
    # axs[1].set_ylabel('$P(n)$')
    # axs[2].plot(n, vPss(n,x3,y3,z3), label='Analytic')
    # axs[2].plot(n, P_normal(n,x3,y3,z3), label='Normal') 
    # # plt.plot(n, vPss(n,x1,y1,z1), label='Analytic')
    # # plt.plot(n, P_normal(n,x1,y1,z1), label='Normal')
    # # plt.legend()
    # plt.xlabel('$n$')
    # plt.tight_layout()
    # plt.savefig('distributions')
    
    x,y,z = 5.1, 5.1, 5.1
    n= np.arange(0,y*2)
    plt.plot(n,vPss(n,x,y,z))
    plt.plot(n, P_normal(n,x,y,z), label='Normal')
    plt.show()
    
    # Fisher_ns(10,2,3,4,'acc')
    # print(Fisher_nsg_num(1,1,1,1,'spec'), Fisher_ns(1, 1, 1, 1,'spec'))
    
    # print(aFisher(x,y,z,'spec'),rel_e(x,y,z,'spec'))
    # vf = np.vectorize(Terms_mp)
    # plt.plot(z,vf(T,x,y,z), label='good')
    # plt.plot(z, Terms_n(T,x,y,z), label='numerical')
    # plt.legend()
    # plt.show()

    # plt.plot(z, vm(T,x,y,z), label='Diff')
    # plt.plot(z, ns_var2(T,x,y,z), label='Manual')
    # plt.legend()
