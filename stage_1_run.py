# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 21:12:09 2019

Creates the plausible prior samples (stage 1, beta 0)
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
from Gorkhamakemesh import *
from greenfunction import * 
from posteriorGorkha import *
import time

#%% Define functions

def lin_inv(model, subdisp, subloc, sublos, W, musq, disct_x, disct_z, surf_pts, LB, UB):
    '''
    Regularize non-negative linear least squares slip inversion RNNLSQ
    '''
    bestgeo = model[:10]
    NT1 = namedtuple('NT1', \
                     'trired p q r xfault yfault zfault disct_x disct_z surfpts model')
    mesh = NT1(None, None, None, None, None, None, None, \
               disct_x, disct_z, surf_pts, bestgeo)
            
    finalmesh = Gorkhamesh(mesh, NT1)
    grn1, obsdata = grn_func(subloc, subdisp, sublos, finalmesh.trired, \
                             finalmesh.p, finalmesh.q, finalmesh.r)
    
    obsdata = np.reshape(obsdata, (obsdata.shape[0], 1))

    numpars = 2*finalmesh.trired.shape[0]
    numdata = subdisp.shape[0]
    greens1 = grn1
    lb = LB[-numpars:]; ub = UB[-numpars:]
    
    laplac1 = laplacian(finalmesh.trired, finalmesh.p, finalmesh.q, finalmesh.r)
    
    laplac = np.r_[np.c_[laplac1.todense()*musq, np.zeros(laplac1.todense().shape)], \
                         np.c_[np.zeros(laplac1.todense().shape), laplac1.todense()*musq]]
    
    Amat = np.r_[np.matmul(W, greens1), laplac]
    Bmat = np.r_[np.matmul(W, obsdata), np.zeros((numpars, 1))]
    
    tikhA = np.matmul(np.transpose(Amat), Amat) + \
            .005**2* np.eye(Amat.shape[1])
    tikhA = np.array(tikhA)
            
    tikhB = np.matmul(np.transpose(Amat), Bmat)
    tikhB = np.array(tikhB).flatten('F')
       
    res = lsq_linear(tikhA, tikhB, bounds=(lb, ub), method = 'trf', \
                     lsmr_tol='auto', max_iter= 500, verbose=1)
    
    return res

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
diffbnd = UB - LB
diffbndN = np.tile(diffbnd, (Nmarkov, 1))
LBN = np.tile(np.transpose(LB), (Nmarkov, 1))
randadd = np.random.rand(Nmarkov, LB.shape[0])
#samplestage = LBN + randadd*diffbndN
#sio.savemat('sample0stage.mat', {'samplestage':samplestage})
mat_samplestage = sio.loadmat('samples1/stage1/sample1stage.mat')
samplestage = mat_samplestage['samplestage']

#indarray = np.int(os.environ['arrayindex'])
#numind = 8; 
#index = np.arange((indarray-1)*numind, indarray*numind)

# %%
# sliparray = np.zeros((numind, 2*finalmesh.trired.shape[0]))
# for i in index:
#     j = np.where(index == i)
#     arraymodel = samplestage[i, :]
#     slip = lin_inv(arraymodel, subdisp, subloc, sublos, W, .4, disct_x, disct_z, surf_pts, \
#                    LB, UB)
#     slipall = slip.x 
#     sliparray[j,:] = slipall

# sampstage = np.c_[samplestage[index,:11], sliparray]

# varname = 'samples1/stage1/samples/sample1stage' + np.str(indarray) + '.mat'
# sio.savemat(varname, {'sampstage':sampstage})
#%%

postval = np.zeros((4000, 1))
for i in range(4000):
    print(i)
    sample_i = samplestage[i,:]
    postsamp = postGorkha(sample_i)
    postval[i] = postsamp
    
beta = np.array([0]) 
stage = np.array([1]) 

NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')
samples = NT2(samplestage, postval, beta, stage, None, None)

varname = 'samples1/stage1/sample1stage.mat'    
sio.savemat(varname, {'samplestage':samples.allsamples, 'postval':samples.postval, \
                     'beta':samples.beta, 'stage':samples.stage})
    





