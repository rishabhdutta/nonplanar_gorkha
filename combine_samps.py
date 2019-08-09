#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:46:57 2019
combines the samples 

@author: duttar
"""
import numpy as np
import scipy.io as sio

mat_resmp = sio.loadmat('resampstage.mat')
stage = mat_resmp['stage']
stagenow = stage[-1][-1]

numind = 2

samp_all = np.zeros((4000, 971))
post_all = np.zeros((4000, 1))
for i in range(2000):
    sampname = 'samples/sample'+ np.str(stagenow) +'stage' + np.str(i+1) + '.mat'
    mat_c1 = sio.loadmat(sampname)
    
    index = np.arange(i*numind, (i+1)*numind)
    sampsnow = mat_c1['samplestage']
    postnow = mat_c1['postval']
    beta = mat_c1['beta']
    stage = mat_c1['stage']
    
    samp_all[index, :] = sampsnow
    post_all[index] = postnow
    
varname = 'sample' + np.str(stagenow) + 'stage.mat'
sio.savemat(varname, {'samplestage':samp_all, 'postval':post_all, \
            'stage':stage, 'beta':beta})

    
    
    
    
    
    
    
    
    
    







