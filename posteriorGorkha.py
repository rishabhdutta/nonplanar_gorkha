#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:09:03 2019

@author: duttar
"""
import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.sparse import lil_matrix
import sys
sys.path.append('additional_scripts/geompars/')
sys.path.append('additional_scripts/greens/')
from Gorkhamakemesh import *
from greenfunction import * 
from collections import namedtuple

def calc_moment(trired, p, q, r, slipall):
    '''
    calculates the moment magnitude for the non-planar fault  
    '''
    
    N = trired.shape[0]
    moment = np.array([])
    for i in range(N):
        ind1 = trired[i,:]
        ind = ind1.astype(int)
        x = p[ind]
        y = q[ind]
        z = r[ind]
        ons = np.array([1,1,1])
        xymat = np.vstack((x,y,ons))
        yzmat = np.vstack((y,z,ons))
        zxmat = np.vstack((z,x,ons))
        detxy = np.linalg.det(xymat)
        detyz = np.linalg.det(yzmat)
        detzx = np.linalg.det(zxmat)
        A = 0.5*np.sqrt(detxy**2+detyz**2+detzx**2)
        Area = A*1e6
        slip1dip = slipall[i]
        slip2strike = slipall[N+i]
        slip = np.abs(slip1dip) + np.abs(slip2strike)
        moment = np.append(moment,3e10*Area*slip)
        
    tot_mom = moment.sum(axis=0)
    momentmag = 2*math.log10(tot_mom)/3 - 6.03 
    return momentmag

def laplacian(trired, p, q, r):
    '''
    Laplacian for triangular dislocation elements for either strike-slip or dip-slip  
    Inputs:   trired - indices for the fault with TDEs
              p,q,r - parameters for the location of TDEs 
    Outputs: laplac  
    '''
    npat = trired.shape[0]
    laplac = lil_matrix((npat,npat))
    
    for i in range(1,npat+1):   
        # 3 corners of ith patch 
        indi1 = trired[i-1,:]
        indi = indi1.astype(int)
        centr_i = np.array([np.mean(p[indi]),np.mean(q[indi]),np.mean(r[indi])])

        # now find the 3 triangles sharing the edges 

        # 1st edge is in following patches 
        firedge,trash = np.where(trired == indi[0])

        # 2nd edge is in following patches
        secedge,trash = np.where(trired == indi[1])

        # 3rd edge is in following patches:
        thiedge,trash = np.where(trired == indi[2])

        # find the triangle sharing 1st and 2nd corners 
        comm12 = np.intersect1d(firedge,secedge)
        indkeep = np.where(comm12!=i-1)
        tri12 = comm12[indkeep]

        # find the triangle sharing 1st and 2nd corners 
        comm23 = np.intersect1d(secedge,thiedge)
        indkeep = np.where(comm23!=i-1)
        tri23 = comm23[indkeep]

        # find the triangle sharing 1st and 2nd corners 
        comm31 = np.intersect1d(firedge,thiedge)
        indkeep = np.where(comm31!=i-1)
        tri31 = comm31[indkeep]

        tris = np.array([tri12,tri23,tri31])
        tris = np.array([item for item in tris if item.size])
        numtris = tris.size

        if numtris == 3:
            # center of 1st triangle: 
            indvert1 = trired[tris[0],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_fir = np.array([centr_x,centr_y,centr_z])
            distri1 = np.sqrt((centr_fir[0]-centr_i[0])**2 + (centr_fir[1]-centr_i[1])**2 + \
            (centr_fir[2]-centr_i[2])**2)

            # center of 2nd triangle
            indvert1 = trired[tris[1],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_sec = np.array([centr_x,centr_y,centr_z])
            distri2 = np.sqrt((centr_sec[0]-centr_i[0])**2 + (centr_sec[1]-centr_i[1])**2 + \
            (centr_sec[2]-centr_i[2])**2)

            # center of 3rd triangle
            indvert1 = trired[tris[2],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_thi = np.array([centr_x,centr_y,centr_z])
            distri3 = np.sqrt((centr_thi[0]-centr_i[0])**2 + (centr_thi[1]-centr_i[1])**2 + \
            (centr_thi[2]-centr_i[2])**2)

            laplac[i-1,tris[0]] = -distri2*distri3
            laplac[i-1,tris[1]] = -distri1*distri3
            laplac[i-1,tris[2]] = -distri1*distri2

            laplac[i-1,i-1] = distri2*distri3 + distri1*distri3 + distri1*distri2

        elif numtris == 2:

            # center of 1st triangle: 
            indvert1 = trired[tris[0],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_fir = np.array([centr_x,centr_y,centr_z])
            distri1 = np.sqrt((centr_fir[0]-centr_i[0])**2 + (centr_fir[1]-centr_i[1])**2 + \
            (centr_fir[2]-centr_i[2])**2)

            # center of 2nd triangle
            indvert1 = trired[tris[1],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_sec = np.array([centr_x,centr_y,centr_z])
            distri2 = np.sqrt((centr_sec[0]-centr_i[0])**2 + (centr_sec[1]-centr_i[1])**2 + \
            (centr_sec[2]-centr_i[2])**2)

            laplac[i-1,tris[0]] = -distri1*distri2
            laplac[i-1,tris[1]] = -distri1*distri2

            laplac[i-1,i-1] = distri1*distri2 + distri1*distri2

        elif numtris == 1:
            # center of 1st triangle: 
            indvert1 = trired[tris[0],:]
            indvert = indvert1.astype(int)
            centr_x = np.mean(p[indvert],axis=1)
            centr_y = np.mean(q[indvert],axis=1)
            centr_z = np.mean(r[indvert],axis=1)
            centr_fir = np.array([centr_x,centr_y,centr_z])
            distri1 = np.sqrt((centr_fir[0]-centr_i[0])**2 + (centr_fir[1]-centr_i[1])**2 + \
            (centr_fir[2]-centr_i[2])**2)

            laplac[i-1,tris[0]] = -distri1*distri1
            laplac[i-1,i-1] = distri1*distri1

    return laplac  

# %% 
class posteriorGorkha:
    '''
    generates the posterior class with different functions to calculate
    a. prior probabilities
    b. likelihood function 
    '''
    def __init__(self, optall, NT1, output, NT2, verbose=False):
        """
        Parameters input: 
            optall : named tuple 
                - optall.surf_pts
                - optall.disct_x
                - optall.disct_z
                - optall.LB
                - optall.UB
                - optall.subdisp
                - optall.subloc
                - optall.sublos
                - optall.W
                                                          
            NT1: create optall object
            
        Parameters output: 
            output : named tuple 
                - output.logfinal
                - output.reslaplac
                - output.resdata
                - output.momag
                - output.trired
                - output.p
                - output.q
                - output.r
                - output.xfault
                - output.yfault
                - output.zfault
                 
            NT2: create opt object
            
        written by: Rishabh Dutta, Jul 11 2019
        (Don't forget to acknowledge)
        
        """
        self.verbose = verbose
        self.optall = optall
        self.NT1 = NT1
        self.output = output
        self.NT2 = NT2 
        
    def initialize(self):
        if self.verbose:
            print ("-----------------------------------------------------------------------------------------------")
            print ("-----------------------------------------------------------------------------------------------")
    
    def slip_prior(self, model):
        '''
        calulate the slip prior probability using the laplacian function 
        '''
        numslip = self.output.trired.shape[0]
        numgeo = model.shape[0] - 2*numslip
        slipall = model[numgeo:]
        slipall = slipall.flatten('F')
        
        laplac1 = laplacian(self.output.trired, self.output.p, self.output.q, \
                            self.output.r)
        laplac = np.r_[np.c_[laplac1.todense(), np.zeros(laplac1.todense().shape)], \
                             np.c_[np.zeros(laplac1.todense().shape), laplac1.todense()]]
        lapslip = np.matmul(laplac, slipall)
        ltl = np.matmul(np.transpose(laplac), laplac)
        
        prior1 = np.matmul(np.matmul(np.transpose(slipall), ltl), slipall)
        return prior1, lapslip, ltl
    
    def likelihood(self, model, greens):
        '''
        Calculates the likelihood function using the greens function
        '''
        numslip = self.output.trired.shape[0]
        numgeo = model.shape[0] - 2*numslip
        slipall = model[numgeo:]
        slipall = slipall.flatten('F')
         
        preddata = np.matmul(greens, slipall)
        datavector = self.optall.subdisp.flatten('F')
        error = preddata - datavector 
        weighterror = np.matmul(self.optall.W, error)
        objfn = np.matmul(np.transpose(weighterror), weighterror)
         
        return objfn, error
                 
        
   # %%    
def posterior(model, optall, NT1, output, NT2):
    '''
    Calculates the logposterior values 
    '''
    
    output = NT2(-np.inf, None, None, None, None, None, None, None, None, None, \
                 None)
    
    # check if model within bounds ####################
    ltmod = np.where((model <= optall.UB))
    if ltmod[0].shape[0] != optall.UB.shape[0]:
        print('lower bound not satisfied \n')
        return output 
    
    mtmod = np.where((model >= optall.LB))
    if mtmod[0].shape[0] != optall.LB.shape[0]:
        print('upper bound not satisfied \n')
        return output
    ###################################################
    
    # get the mesh ####################################
    NTmesh = namedtuple('NTmesh', 'trired p q r xfault yfault zfault disct_x disct_z surfpts model')
    mesh = NTmesh(None, None, None, None, None, None, None, optall.disct_x, \
                  optall.disct_z, optall.surf_pts, model)
            
    finalmesh = Gorkhamesh(mesh, NTmesh) 
    output = NT2(-np.inf, None, None, None, finalmesh.trired, finalmesh.p, \
                 finalmesh.q, finalmesh.r, finalmesh.xfault, finalmesh.yfault, \
                 finalmesh.zfault)
    
    # check if the fault is folding upwards
    inall_z = np.zeros((finalmesh.zfault.shape[0]-1, finalmesh.zfault.shape[1]))
    for i in range(finalmesh.zfault.shape[1]):
        check_z = finalmesh.zfault[:,i]
        in_z = np.zeros((finalmesh.zfault.shape[0]-1, 1))
        for j in range(1,check_z.shape[0]):
            if check_z[j] > check_z[j-1]:
                in_z[j-1] = 1
            else:
                in_z[j-1] = 0
        inall_z[:,i] = in_z.flatten('F')
        
    if np.any(inall_z == 1) == True:
        print('the fault is folding upwards \n')
        print(inall_z)
        return output
    
    ###################################################
    
    # set the slip values #############################
    numslip = finalmesh.trired.shape[0]
    numgeo = model.shape[0] - 2*numslip
    slipall = model[numgeo:]
    dipslip = model[numgeo:numgeo+numslip]
    strikeslip = model[-numslip:]
    
    # check the moment magnitude ######################
    momag = calc_moment(finalmesh.trired, finalmesh.p, finalmesh.q, \
                        finalmesh.r, slipall)
    
    if momag > 8.3 or momag < 7.3: 
        print('momeng magnitude is spurious \n')
        print(momag)
        return output
    
    output = NT2(-np.inf, None, None, momag, finalmesh.trired, finalmesh.p, \
                 finalmesh.q, finalmesh.r, finalmesh.xfault, finalmesh.yfault, \
                 finalmesh.zfault)
    
    ###################################################
    
    # define the hyperparameters ######################
    sigmasq = 10**model[10]
    alphasq = 10**model[9]
    
    musq = sigmasq/alphasq
    
    # calculate the slip prior probability ############
    current = posteriorGorkha(optall, NT1, output, NT2)
    objfnslip, reslaplac, ltl = current.slip_prior(model)
    
    output = NT2(-np.inf, reslaplac, None, momag, finalmesh.trired, finalmesh.p, \
                 finalmesh.q, finalmesh.r, finalmesh.xfault, finalmesh.yfault, \
                 finalmesh.zfault)
    
    try:
        invltlval = np.sum(np.log10(np.diag(np.linalg.cholesky(ltl))))
    except:
        print('Laplacian is not positive definite. So checking the determinant directly \n')  
        return output    
    
    # calculate the greens function ###################
    grn1, obsdata = grn_func(optall.subloc, optall.subdisp, optall.sublos, \
                             finalmesh.trired, finalmesh.p, finalmesh.q, \
                             finalmesh.r)
    
    # calculate the likelihood function ###############
    current = posteriorGorkha(optall, NT1, output, NT2)
    objfnlikeli, resdata = current.likelihood(model, grn1)
    
    output = NT2(-np.inf, reslaplac, resdata, momag, finalmesh.trired, finalmesh.p, \
                 finalmesh.q, finalmesh.r, finalmesh.xfault, finalmesh.yfault, \
                 finalmesh.zfault)
    
    # calculate the normalization factor
    MLnum = 2* finalmesh.trired.shape[0]
    Nnum = optall.subdisp.shape[0]
    
    first = -MLnum/2*np.log10(musq) - Nnum/2*np.log10(sigmasq) + invltlval
    logfinal = first - (1/(2*sigmasq))*(objfnlikeli  +alphasq* objfnslip)/np.log10(np.exp(1))
    logfinal = np.asarray(logfinal).reshape(-1)
    
    output = NT2(logfinal, reslaplac, resdata, momag, finalmesh.trired, finalmesh.p, \
                 finalmesh.q, finalmesh.r, finalmesh.xfault, finalmesh.yfault, \
                 finalmesh.zfault)
    
    return output

    
    
    

