import numpy as np
from pyhyp import pyHyp
import os

data = np.loadtxt('new.dat')
x=data[:,0].copy()
newY=data[:,1].copy()
ndim=x.shape[0]

airfoil3d = np.zeros((ndim,2,3))
for j in xrange(2):
    airfoil3d[:,j,0] = x[:]
    airfoil3d[:,j,1] = newY[:]
airfoil3d[:,0,2] = 0.0
airfoil3d[:,1,2] = 1.0
#write out plot3d
fsam = open('new.xyz','w')
fsam.write(str(1)+'\n')
fsam.write(str(ndim)+' '+str(2)+' '+str(1)+'\n')
for l in xrange(3):
    for k in xrange(1):
        for j in xrange(2):
            for i in xrange(ndim):
                fsam.write('%.15f\n'%(airfoil3d[i,j,l]))
fsam.close()

options= {
    # ---------------------------
    #        Input Parameters
    # ---------------------------
    'inputFile':'new.xyz',
    'unattachedEdgesAreSymmetry':False,
    'outerFaceBC':'farField',
    'autoConnect':True,
    'BC':{1:{'jLow':'zSymm',
             'jHigh':'zSymm'}},
    'families':'wall',

    # ---------------------------
    #        Grid Parameters
    # ---------------------------
    'N': 129,
    's0':3e-6,
    'marchDist':100,
    
    # ---------------------------
    #   Pseudo Grid Parameters
    # ---------------------------
    'ps0':-1,
    'pGridRatio':-1,
    'cMax':3.0,

    # ---------------------------
    #   Smoothing parameters
    # ---------------------------
    'epsE': 1.0,
    'epsI': 2.0,
    'theta': 3.0,
    'volCoef': .25,
    'volBlend': 0.0001,
    'volSmoothIter': 100,
}


hyp = pyHyp(options=options)
hyp.run()
hyp.writeCGNS('baseairfoil.cgns')
