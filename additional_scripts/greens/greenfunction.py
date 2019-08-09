#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 22:13:10 2019

This greens function code is only for InSAR data.
@author: duttar
"""
import numpy as np
from tde import * 

def grn_func(subcoord, subdata, sublos, trired, p, q, r):
    '''
    Calculates the greens function for triangular dislocation elements TDEs using Meade's triangulation code  
    Inputs: subcoord - coordinates of geodetic data
            subdata - geodetic data 
            sublos - line of sight of the data points
            trired - indices for the fault with TDEs
            p,q,r - parameters for the location of TDEs 
    Outputs: greens - green's function matrix for dip-slip and strike-slip 
             datavector - data vector 
    '''
    numdata = np.int(np.size(subdata))
    numpars = 2* trired.shape[0]
    
#    trired1 = trired.flatten('F')
#    p_tri = np.array([])
#    q_tri = np.array([])
#    r_tri = np.array([])
#    num_int = np.int(3*numpars/2)
#    for i in range(num_int): 
#        trired2 = np.int(trired1[i])
#        p_tri = np.append(p_tri,p[trired2-1])
#        q_tri = np.append(q_tri,q[trired2-1])
#        r_tri = np.append(r_tri,-r[trired2-1])
    p_tri = p[trired]; q_tri = q[trired]; r_tri = -r[trired]
    
    num_int = np.int(numpars/2)

    xcoord = subcoord[:,0]
    ycoord = subcoord[:,1]
    zcoord = -subcoord[:,2]
    
    greens1 = np.zeros((numdata,num_int))
    greens2 = np.zeros((numdata,num_int))
    
#    start = time.time()
    for i in range(num_int):
#        start = time.time()
        xparco = p_tri[i,:]
        yparco = q_tri[i,:]
        zparco = r_tri[i,:]
        Uall1 = calc_tri_displacements(xcoord, ycoord, zcoord, xparco, yparco, zparco, .28, 0, 0, -1)
        
        dataunit1 = np.c_[Uall1['x'],Uall1['y'],-Uall1['z']]
        dataunit2 = dataunit1 * sublos
        dataunit3 = dataunit2.sum(axis =1)
        greens1[:,i] = dataunit3
        
        Uall2 = calc_tri_displacements(xcoord, ycoord, zcoord, xparco, yparco, zparco, .28, 1, 0, 0)
        dataunit4 = np.c_[Uall2['x'],Uall2['y'],-Uall2['z']]
        dataunit5 = dataunit4 * sublos  
        dataunit6 = dataunit5.sum(axis =1)
        greens2[:,i] = dataunit6 
#        end = time.time()
#        print(end - start)
    
#    end = time.time()
#    print(end - start)
    
    greens = np.c_[greens1,greens2]
    datavector = subdata.flatten('F')
    
    return greens, datavector


