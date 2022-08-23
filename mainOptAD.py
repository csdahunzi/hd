#!/usr/bin/env python
# coding: utf-8
'''
Author: Bouhlel Mohamed Amine

This script runs the optimization of airfoil shape analysis test case in subsonic regime for the paper:
Scalable gradient-enhanced artificial neural networks for airfoil shape design in the subsonic and transonic regimes

To run this script, the user should adapt the file paths
'''

import numpy as np
import tensorflow as tf
from pyoptsparse import Optimization, OPT
from scipy.misc import derivative
from scipy import interpolate
import os

def predict(x):
    x = np.hstack(((x[:-1]).reshape((1,-1)),0.45*np.ones((1,1)),x[-1]*np.ones((1,1))))
    np.savetxt('sample.txt',x)
    os.system('python run.py')
    res = np.loadtxt('results.txt')
    return res[0], res[1],res[2:17],res[17:]


naca0012 = np.loadtxt('naca0012.dat').reshape((1,-1))
naca0012 = np.hstack((naca0012,1*np.ones((1,1))))

nc = 7
nt = 7

xslc = np.loadtxt('./airfoil/db/subsonic_grid/xslice.txt')
yslc = np.loadtxt('./airfoil/db/subsonic_grid/yslice.txt')

boundupper=[]
boundlower=[]
boundupper.append('none')
boundlower.append('none')
for inc in xrange(1,nc+nt):
    if inc==nc:
        boundupper.append('none')
        boundlower.append('none')
        continue
    zupper = np.loadtxt('./airfoil/db/subsonic_grid/zupper'+str(inc)+'.txt')
    zlower = np.loadtxt('./airfoil/db/subsonic_grid/zlower'+str(inc)+'.txt')
    tupper = interpolate.interp2d(xslc, yslc, zupper.transpose(), kind='cubic')#kind='quintic')
    tlower = interpolate.interp2d(xslc, yslc, zlower.transpose(), kind='cubic')#kind='quintic')
    boundupper.append(tupper)
    boundlower.append(tlower)

def upperfunc_obj(i,t1,c1):
    myt1 = t1
    myc1 = c1
    if i==0 or i==nc:
        return 0.0
    else:
        myfunc = boundupper[i]([myt1],[myc1])
        return myfunc[0]

# func value of upper surface i
def lowerfunc_obj(i,t1,c1):
    myt1 = t1
    myc1 = c1        
    if i==0 or i==nc:
        return 0.0
    else:
        myfunc = boundlower[i]([myt1],[myc1])
        return myfunc[0]

def upperfunc_grad(i,t1,c1,nflow=1):
        
    grad=np.zeros((1,nc+nt+nflow))
    myfunc = np.zeros(2)
    if i==0 or i==nc:
        return grad
    else:
        fdstep = 1.e-6
        myc1 = c1
        myt0 = t1 - fdstep
        myt2 = t1 + fdstep
        myfunc[0] = boundupper[i]([myt0],[myc1])[0]
        myfunc[1] = boundupper[i]([myt2],[myc1])[0]
        dfdt = (myfunc[1] - myfunc[0])/(2.0*fdstep)

        myc0 = c1 - fdstep
        myc2 = c1 + fdstep
        myt1 = t1
        myt1 = t1

        myfunc[0] = boundupper[i]([myt1],[myc0])[0]
        myfunc[1] = boundupper[i]([myt1],[myc2])[0]
        dfdc = (myfunc[1] - myfunc[0])/(2.0*fdstep)

        grad[0,i] = 1.0
        grad[0,nc] = -1.0*dfdt
        grad[0,0] = -1.0*dfdc

        return grad

def lowerfunc_grad(i,t1,c1,nflow=1):
    grad=np.zeros((1,nc+nt+nflow))
    myfunc = np.zeros(2)
    if i==0 or i==nc:
        return grad
    else:
        fdstep = 1.e-6
        myc1 = c1
        myt0 = t1 - fdstep
        myt2 = t1 + fdstep
        myfunc[0] = boundlower[i]([myt0],[myc1])[0]
        myfunc[1] = boundlower[i]([myt2],[myc1])[0]
        dfdt = (myfunc[1] - myfunc[0])/(2.0*fdstep)

        myc0 = c1 - fdstep
        myc2 = c1 + fdstep
        myt1 = t1

        myfunc[0] = boundlower[i]([myt1],[myc0])[0]
        myfunc[1] = boundlower[i]([myt1],[myc2])[0]
        dfdc = (myfunc[1] - myfunc[0])/(2.0*fdstep)

        grad[0,i] = 1.0
        grad[0,nc] = -1.0*dfdt
        grad[0,0] = -1.0*dfdc

        return grad


def objfunc(xdict):
    x = xdict['xvars']
    funcs = {}
    cd, cl,dcd,dcl = predict(x)
    funcs['obj'] = cd
    funcs['con1'] = cl

    # Constraints on higer-order modes
    myt = x[nc]
    myc = x[0]
    for i in xrange(1,nc):
        funcs['mode_uppercon_'+str(i)] = x[i] - upperfunc_obj(i,myt,myc)
        funcs['mode_lowercon_'+str(i)] = x[i] - lowerfunc_obj(i,myt,myc)
    for i in xrange(1,nc):
        funcs['mode_uppercon_'+str(i+nc)] = x[i+nc] - upperfunc_obj(i+nc,myt,myc)
        funcs['mode_lowercon_'+str(i+nc)] = x[i+nc] - lowerfunc_obj(i+nc,myt,myc)
    
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    x = xdict['xvars']
    funcsSens = {}
    cd, cl,dcd,dcl = predict(x)
    funcsSens['obj', 'xvars'] = dcd
    funcsSens['con1', 'xvars'] = dcl

    # Constraints on higer-order modes
    myt = x[nc]
    myc = x[0]
    for i in xrange(1,nc):
        funcsSens['mode_uppercon_'+str(i),'xvars'] = upperfunc_grad(i,myt,myc)
        funcsSens['mode_lowercon_'+str(i),'xvars'] = lowerfunc_grad(i,myt,myc)
    for i in xrange(1,nt):
        funcsSens['mode_uppercon_'+str(i+nc),'xvars'] = upperfunc_grad(i+nc,myt,myc)
        funcsSens['mode_lowercon_'+str(i+nc),'xvars'] = lowerfunc_grad(i+nc,myt,myc)

    fail = False
    return funcsSens, fail

# Optimization Object
optProb = Optimization('Airfoil shape optimization', objfunc)

base = np.loadtxt('db_modescoef.txt')
# Design Variables
lower = []
for i in range(14):
    lower.append(np.min(base[:,i]))
lower.append(-.5)

upper = []
for i in range(14):
    upper.append(np.max(base[:,i]))
upper.append(5)

upper[7] = .9 * naca0012[0,7]
lower[7] = upper[7] - 0.3

value = naca0012

optProb.addVarGroup('xvars', 15, lower=lower, upper=upper, value=value)
# Constraints
optProb.addCon('con1', lower=0.5, upper=0.5)

# adding mode constriaints to exclude abnormal airfoils
for i in xrange(1,nc):
    optProb.addCon('mode_uppercon_'+str(i), lower=-100.0, upper=0.0, scale=1.0)
    optProb.addCon('mode_lowercon_'+str(i), lower=0.0,    upper=100.0, scale=1.0)
for i in xrange(1,nt):
    optProb.addCon('mode_uppercon_'+str(i+nc), lower=-100.0, upper=0.0, scale=1.0)
    optProb.addCon('mode_lowercon_'+str(i+nc), lower=0.0,    upper=100.0, scale=1.0)

# Objective
optProb.addObj('obj')

# Check optimization problem:
#print(optProb)

# Optimizer

opt = OPT('SNOPT')

# Solution
if 0:
    histFileName = '%s_hs015_Hist.hst'%(args.opt.lower())
else:
    histFileName = None
# end

sol = opt(optProb, sens=sens,storeHistory=histFileName)

print sol
# Check Solution
print sol.xStar['xvars']
np.savetxt('sol.dat',(sol.xStar['xvars']).reshape((1,-1)))
print sol.fStar
