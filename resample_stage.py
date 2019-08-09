#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:34:53 2019

Resampling step of SMC sampling for Gorkha case
@author: duttar
"""

# %%
# load libraries 
import numpy as np
import scipy.io as sio
import random
import sys
import os
from scipy.optimize import lsq_linear
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#simport matplotlib.tri as mtri
from collections import namedtuple
sys.path.append('additional_scripts/')
sys.path.append('additional_scripts/geompars/')
sys.path.append('additional_scripts/greens/')
sys.path.append('additional_scripts/SMC-python/')
from Gorkhamakemesh import *
from greenfunction import * 
from posteriorGorkha import *
from SMC import *
import time

#%% Define functions


# %%

mat_c1 = sio.loadmat('additional_scripts/GPS_subsampledGorkha.mat')
covall = mat_c1['covall']
subdisp = mat_c1['subdisp']
subloc1 = mat_c1['subloc']
sublos = mat_c1['sublos']
numdis = subloc1.shape[0]

subloc = np.hstack((subloc1,np.zeros((numdis,1))))

for i in range(numdis):
    covall[i,i] = 1.19*covall[i,i]

invcov = np.linalg.inv(covall)
W = np.linalg.cholesky(invcov).T

surf_pts = np.array([[215.8972,3.0950e+03],[442.8739,3.0097e+03]])
bestgeo = np.array([1.0927 ,0.0241, 5.6933,-20.0000, 2.1345, \
                    0.3529, 4.4643, 0.0336, -12, -2.8494, 0.0043])
disct_x = 20; disct_z = 12

NT1 = namedtuple('NT1', \
'trired p q r xfault yfault zfault disct_x disct_z surfpts model')
mesh = NT1(None, None, None, None, None, None, None, \
            disct_x, disct_z, surf_pts, bestgeo)
            
# get the geometry 
start = time.time()
finalmesh = Gorkhamesh(mesh, NT1)          
end = time.time()
print(end - start)

################# make a plot #################################
#fig = plt.figure(figsize=plt.figaspect(0.5))

#ax = fig.add_subplot(1, 1, 1, projection='3d')
#ax.plot_trisurf(finalmesh.p, finalmesh.q, finalmesh.r, \
#                triangles=finalmesh.trired, cmap=plt.cm.Spectral)
#plt.axis('equal')
#ax.set(xlim=(200, 500), ylim=(2900, 3400), zlim=(-25, -5))
#plt.show()
##############################################################

# run the greens function 
start = time.time()
#grn1, obsdata = grn_func(subloc, subdisp, sublos, finalmesh.trired, \
#                         finalmesh.p, finalmesh.q, finalmesh.r)
end = time.time()
print(end - start)

##############################################################

lowslip1 = np.zeros((1,finalmesh.trired.shape[0]))
lowslip2 = -10*np.ones((1,finalmesh.trired.shape[0]))
maxslip1 = 25*np.ones((1,finalmesh.trired.shape[0]))
maxslip2 = 10*np.ones((1,finalmesh.trired.shape[0]))

LBslip = np.append(lowslip1, lowslip2)
UBslip = np.append(maxslip1, maxslip2)

LB = np.append(np.array([-5, -.5, 3, -25, -5, -.5, -5, -.5, -13, -8, -8]),LBslip)
UB = np.append(np.array([7, .5, 25, -16, 7, .5, 7, .5, -6,  8,  8]),UBslip)

edges = np.array([], dtype = int)
for i in range(1, disct_x+1):
    if i == 1:
        edges = np.append(edges, np.arange(1,(disct_z)*2+1))
    elif i < disct_x and i > 1:
        edges = np.append(edges, np.array([(i-1)*disct_z*2+1, (i-1)*disct_z*2+2, \
                                           i*(disct_z)*2, i*(disct_z)*2-1]))
    elif i == disct_x:
        edges = np.append(edges, np.arange((i-1)*(disct_z)*2+1,(i)*(disct_z)*2+1))
edges = edges-1
LB[edges+bestgeo.shape[0]] = 0
UB[edges+bestgeo.shape[0]] = 1e-5
LB[edges+bestgeo.shape[0]+finalmesh.trired.shape[0]] = 0
UB[edges+bestgeo.shape[0]+finalmesh.trired.shape[0]] = 1e-5
        
NTpostin = namedtuple('NTpostin', \
                      'surf_pts disct_x disct_z LB UB subdisp subloc sublos W')            
optall = NTpostin(surf_pts, disct_x, disct_z, LB, UB, subdisp, subloc, \
                  sublos, W)

NTpostout = namedtuple('NTpostout', \
                       'logpost, reslaplac, resdata, momag, trired, p, q, r, xfault, yfault, zfault')
output = NTpostout(None, None, None, None, None, None, None, None, None, None, None)

def postGorkha(x):
    postout = posterior(x, optall, NTpostin, output, NTpostout)
    return postout.logpost

################################################################

Nmarkov = 4000; Nchains = 100

varname1 = 'samples1/stage10/sample10stage.mat'
mat_samplestage = sio.loadmat(varname1)
samplestage = mat_samplestage['samplestage']
postval = mat_samplestage['postval']
beta = mat_samplestage['beta']
stage = mat_samplestage['stage']

NT1 = namedtuple('NT1', 'N Neff target LB UB')
opt = NT1(Nmarkov, Nchains, postGorkha, LB, UB)

NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl resmplpost')
samples = NT2(samplestage, postval, beta[-1], stage[-1], None, None, None)

current = SMCclass(opt, samples, NT1, NT2)
samples = current.find_beta()            # calculates beta at next stage 
        
current = SMCclass(opt, samples, NT1, NT2)
samples = current.resample_stage()  

current = SMCclass(opt, samples, NT1, NT2)
samples = current.make_covariance()       
        
varname2 = 'samples1/stage11/resampstage.mat'
sio.savemat(varname2, {'samplestage':samples.allsamples, 'postval':samples.postval, \
                     'beta':samples.beta, 'stage':samples.stage, \
                     'covsmpl':samples.covsmpl, 'resmpl':samples.resmpl, \
                     'resmplpost':samples.resmplpost})
    



