import os
import numpy as np

#os.system('mkdir output')
samples = np.loadtxt('sample.txt').reshape(1,-1)
ns = samples.shape[0]
ndim = samples.shape[1]
basedata = np.loadtxt('basis.txt')
x = basedata[0,:].copy()
U = basedata[1:,:].copy()
nbasis = ndim - 2
npts = basedata.shape[1]

for iter in range(ns):
    aoa = samples[iter,-1]
    ma = samples[iter,-2]
    basecoef = samples[iter,:-2].copy()
    f = open('basecoef.txt','w')
    for i in xrange(nbasis):
        f.write('%.15f\n'%(basecoef[i]))
    f.close()
    basey = np.dot(basecoef,U)
    f = open('new.dat','w')
    for i in xrange(npts):
        f.write('%.15f %.15f\n'%(x[i],basey[i]))
    f.close()
    os.system('python genMesh.py')
    os.system('mpirun -np 16 python mainAD.py --alpha {0} | tee readme.log'.format(aoa)) 
    os.system('mv readme.log output/readme{0}.log'.format(iter+1))
    os.system('mv fc_000_vol.cgns output/vol{0}.cgns'.format(iter+1))
    os.system('mv fc_000_slices.dat output/slice{0}.dat'.format(iter+1))
    os.system('mv results.txt output/obj{0}.txt'.format(iter+1))
