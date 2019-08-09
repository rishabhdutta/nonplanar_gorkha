# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import leastsq

def findpoint(p1,p2,somep):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    
    distsq = (x2-x1)**2+(y2-y1)**2
    scale = 2/np.sqrt(distsq) 
    divpt = (y2-y1)/(x2-x1)
    angrot = math.atan(divpt)    # angrot is in radians
    
    if x1 > x2:
        scale = -scale
    
    rotA = np.array([[np.cos(angrot),np.sin(angrot)],[-np.sin(angrot), \
                      np.cos(angrot)]]) 
    trans = scale*np.matmul(rotA,np.reshape(p1,(2,1)))
    
    p = scale * np.matmul(rotA,np.reshape(somep,(2,1))) - trans 
    return p

def reversefindpoint(p1,p2,somep):
    # reverse of this finds somep in the changed refernce CS where p1 p2 on 0 ,2 on xaxis
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    
    distsq = (x2-x1)**2+(y2-y1)**2
    scale = 2/np.sqrt(distsq) 
    divpt = (y2-y1)/(x2-x1)
    angrot = math.atan(divpt)    # angrot is in radians
    
    if x1 > x2:
        scale = -scale
        
    rotA = np.array([[np.cos(angrot),np.sin(angrot)], \
                      [-np.sin(angrot), np.cos(angrot)]])
    trans = scale*np.matmul(rotA,np.reshape(p1,(2,1)))
    
    par1 = np.linalg.inv(rotA)
    par2 = np.reshape(somep,(2,1)) + trans
        
    p = (1/scale)*np.matmul(par1,par2)
    return p

    # %%  
class makemeshclass:
    '''
    generates the mesh class with different functions to create 
    a. curved fault surface 
    b. curved down-dip surface 
    c. differently curved fault bottom 
    '''
    def __init__(self, mesh, NT1, verbose=False):
        """
        Parameters output: 
            mesh : named tuple 
                - mesh.trired (contains the address of the TDEs)
                - mesh.p (all the x-coordinates of the mesh)
                - mesh.q (all the y-coordinates of the mesh)
                - mesh.r (all the z-coordinates of the mesh)
                - mesh.xfault (x-coordinates in order)
                - mesh.yfault (x-coordinates in order)
                - mesh.zfault (x-coordinates in order)
                
                inputs:
                - mesh.disct_x (horizontal discretization)
                - mesh.disct_z (vertical discretization)
                - mesh.surfpts (surface points to start with)
                - mesh.model (mesh model input parameters)
                           
            NT1: create opt object
            
        written by: Rishabh Dutta, Jul 11 2019
        (Don't forget to acknowledge)
        
        """
        self.verbose = verbose
        self.mesh = mesh  
        self.NT1 = NT1
       
    def initialize(self):
        if self.verbose:
            print ("-----------------------------------------------------------------------------------------------")
            print ("-----------------------------------------------------------------------------------------------")
            print(f'Initializing with horizontal discretization: {self.mesh.disct_x :8d} and vertical discretization: {self.mesh.disct_z :8d}.')
                             
    def strike_surface(self):
        '''
        use the geom parameters to determine the curvature of fault top-edge
        '''
        amod1 = self.mesh.model[0]
        amod2_1 = self.mesh.model[1]
        
        if amod1 < 2 and amod1 > 0:
            amod2 = 4.5* amod2_1
        else:
            amod2 = amod2_1
        
        indvert1 = self.mesh.surfpts
        deptop = self.mesh.model[8]
        
        indvert = np.hstack((indvert1,deptop*np.ones([2,1])))
        numdisct = self.mesh.disct_x
        
        p1 = indvert[0,:2]
        p2 = indvert[1,:2]
        
        p1ref = findpoint(p1,p2,p1)
        p2ref = findpoint(p1,p2,p2)
        
        polynom = amod2*np.array([1, -2-amod1, 2*amod1, 0])
        yval = lambda x: polynom[0]*x**3 +polynom[1]*x**2+ polynom[2]*x**1+ \
                    polynom[3]
        
        pol_diff = np.polyder(polynom,1)
        integrad = lambda y: np.sqrt((pol_diff[0]*y**2 + pol_diff[1]*y+ pol_diff[2])**2 +1)
        fulllen_1 = quad(integrad,p1ref[0],p2ref[0])
        fulllen = fulllen_1[0]
        
        leng = lambda x: quad(integrad,p1ref[0],x)
        seglen = np.abs(fulllen/numdisct)
        
        xinc = np.zeros((numdisct+1,1))
        yinc = np.zeros((numdisct+1,1))
        xincreal = np.zeros((numdisct+1,1))
        yincreal = np.zeros((numdisct+1,1))
        
        xinc[0] = p1ref[0]; xinc[-1] = p2ref[0]
        yinc[0] = p1ref[1]; yinc[-1] = p2ref[1]
        xincreal[0] = p1[0]; xincreal[-1] = p2[0]
        yincreal[0] = p1[1]; yincreal[-1] = p2[1]
        for i in range(numdisct-1):
            curvelen = seglen * (i+1)
            val_xinc = leastsq(lambda z: leng(z)-curvelen,1)
            xinc[i+1] = val_xinc[0]
            yinc[i+1] = yval(xinc[i+1])
            
            xy = reversefindpoint(p1,p2,np.array([xinc[i+1],yinc[i+1]]))
            xincreal[i+1] = xy[0]
            yincreal[i+1] = xy[1]
            
        xfault = np.zeros((self.mesh.disct_z+1,self.mesh.disct_x+1))
        yfault = np.zeros((self.mesh.disct_z+1,self.mesh.disct_x+1))
        
        xfault[0,:] = xincreal[:,0]
        yfault[0,:] = yincreal[:,0]
        mesh = self.NT1(None, None, None, None, xfault, yfault, None, \
                        self.mesh.disct_x, self.mesh.disct_z, \
                        self.mesh.surfpts, self.mesh.model)
        return mesh
    
    def downdip_surface(self):
        '''
        use the geom parameters to determine the curvature of fault down-dip
        '''
        meandip = self.mesh.model[2]
        maxdep = self.mesh.model[3]
        tracedep = self.mesh.model[8]
        
        xinc = self.mesh.xfault[0,:]
        yinc = self.mesh.yfault[0,:]
        
        strike1 = np.arctan((yinc[-1] - yinc[0])/(xinc[-1] - xinc[0]))
        strike = strike1*180/np.pi
        
        dipdirn = strike + 90 
        slopem = np.abs(np.tan(dipdirn*np.pi/180))
        
        hori = np.abs(maxdep-tracedep)/np.tan(meandip*np.pi/180)
        
        # find the bottom of the fault 
        numdisct = self.mesh.disct_x
        
        xincbot = np.zeros((numdisct+1,1))
        yincbot = np.zeros((numdisct+1,1))
        for i in range(numdisct+1):
            xtop = xinc[i]; ytop = yinc[i]
            yintc = ytop - slopem*xtop 
            
            polynom = np.array([(1+slopem**2), \
                                (-2*xtop + 2*slopem*yintc - 2*slopem*ytop), \
                                (xtop**2 + yintc**2 + ytop**2 - 2*yintc*ytop - hori**2)])
            
            xvals = np.roots(polynom)
            yvals = slopem * xvals + yintc
            indmore = np.where(xvals > xtop)
            indless = np.where(xvals < xtop)
            
            if meandip < 90: 
                xbot = xvals[indmore]
                ybot = yvals[indmore]
            else :
                xbot = xvals[indless]
                ybot = yvals[indless]
                
            xincbot[i] = xbot; yincbot[i] = ybot
            
        # now find the down-dip curvature 
        bmod1 = self.mesh.model[4]
        bmod2_1 = self.mesh.model[5]
        if bmod1 < 2 and bmod1 > 0:
            bmod2 = 4.5* bmod2_1
        else:
            bmod2 = bmod2_1
        
        zintvert = np.array([tracedep, maxdep])
        
        p1 = np.array([xinc[0], zintvert[0]])
        p2 = np.array([xincbot[0,0], zintvert[1]])
        
        p1ref = findpoint(p1,p2,p1)
        p2ref = findpoint(p1,p2,p2)
        
        polynom = bmod2*np.array([1, -2-bmod1, 2*bmod1, 0])
        zval = lambda x: (polynom[0]*x**3 + polynom[1]*x**2 + polynom[2]*x + \
                          polynom[3])
        pol_diff = np.polyder(polynom) 
        integrad = lambda y : np.sqrt((pol_diff[0]*y**2 + pol_diff[1]*y + \
                                    pol_diff[2])**2 + 1)
        fulllen = quad(integrad, p1ref[0], p2ref[0])
        fulllen = fulllen[0]
        
        leng = lambda x: quad(integrad, p1ref[0], x)
        
        disctz = self.mesh.disct_z
        seglen = np.abs(fulllen/disctz)
        
        xincd = np.zeros((disctz+1,1))
        zincd = np.zeros((disctz+1,1))
        xincdreal = np.zeros((disctz+1,1))
        zincdreal = np.zeros((disctz+1,1))       
        
        xincd[0] = p1ref[0]; xincd[-1] = p2ref[0]
        zincd[0] = p1ref[1]; zincd[-1] = p2ref[1]
        xincdreal[0] = p1[0]; xincdreal[-1] = p2[0]
        zincdreal[0] = p1[1]; zincdreal[-1] = p2[1] 
        
        for i in range(disctz-1): 
            curvelen = seglen * (i+1)
            valleast = leastsq(lambda z: leng(z)- curvelen, 1)
            xincd[i+1,0] = valleast[0]
            zincd[i+1,0] = zval(xincd[i+1])
            
            xy = reversefindpoint(p1,p2,np.array([xincd[i+1], zincd[i+1]]))
            xincdreal[i+1,0] = xy[0]
            zincdreal[i+1,0] = xy[1]
            
        slopem = (yinc[0] - yincbot[0,0])/(xinc[0] - xincbot[0,0])
        xincall = xincdreal - xinc[0]
        
        xfault = np.zeros((self.mesh.disct_z+1,self.mesh.disct_x+1))
        yfault = np.zeros((self.mesh.disct_z+1,self.mesh.disct_x+1))
        zfault = np.tile(zincdreal,(1,numdisct+1))
        
        for i in range(numdisct+1):
            xt = xinc[i]; yt = yinc[i]
            yintc = yt - slopem*xt
            xfault[:,i] = xincall[:,0] + xt
            yfault[:,i] = slopem* xfault[:,i] + yintc 
        mesh = self.NT1(None, None, None, None, xfault, yfault, zfault, \
                        self.mesh.disct_x, self.mesh.disct_z, \
                        self.mesh.surfpts, self.mesh.model)
        return mesh  # check again 
    
    def strike_bottom(self):
        '''
        use the geom parameters to determine the fault bottom curvature
        '''
        numdisct = self.mesh.disct_x
        cmod1 = self.mesh.model[6]
        cmod2_1 = self.mesh.model[7]
        if cmod1 < 2 and cmod1 > 0:
            cmod2 = 4.5* cmod2_1 
        else:
            cmod2 = cmod2_1
            
        xfault = self.mesh.xfault
        yfault = self.mesh.yfault
        
        indvert = np.array([1, numdisct])
        p1 = np.array([xfault[-1,indvert[0]], \
                       yfault[-1,indvert[0]]])
        p2 = np.array([xfault[-1,indvert[1]], \
                       yfault[-1,indvert[1]]])
        
        for i in range(indvert[0]+1, indvert[1]):
            top = np.array([xfault[0,i-1], \
                            yfault[0,i-1]])
            bottom = np.array([xfault[-1,i-1], \
                               yfault[-1,i-1]])
            topref = findpoint(p1, p2, top)
            bottomref = findpoint(p1, p2, bottom)
            
            Apol = (topref[1] - bottomref[1])/(topref[0] - bottomref[0])
            Bpol = topref[1] - Apol * topref[0]
            
            polsolve = np.array([cmod2, -(2+cmod1)*cmod2, \
                                 (cmod1*cmod2*2 - Apol), -Bpol])
            try:
                solx = np.roots(polsolve)
            except:
                solx = np.array([0])
            
            solx[np.where(np.abs(solx- 0) < 1e-10)] = 0
            
            xfault_newref = solx[np.where(np.isreal(solx) == True)]
            xfault_newref = xfault_newref[np.where(xfault_newref >= 0)]
            xfault_newref = xfault_newref[np.where(xfault_newref <= 2)]
            xfault_newref = np.real(np.min(xfault_newref))
            
            yfault_newref = Apol*xfault_newref +Bpol
            
            fault_new = reversefindpoint(p1,p2,np.array([xfault_newref, yfault_newref]))
            xfault_new = np.real(fault_new[0])
            yfault_new = np.real(fault_new[1])
            
            diffx = bottom[0] - xfault_new
            diffy = bottom[1] - yfault_new
            
            xfault[-1,i-1] = xfault_new 
            yfault[-1,i-1] = yfault_new
            for j in range(2, self.mesh.disct_z +1): 
                xfault[j-1,i-1] = (xfault[j-1,i-1] -  \
                                  diffx*(j-2)/(self.mesh.disct_z-1))
                yfault[j-1,i-1] = (yfault[j-1,i-1] -  \
                                  diffy*(j-2)/(self.mesh.disct_z-1))
            
        mesh = self.NT1(None, None, None, None, xfault, yfault, self.mesh.zfault, \
                        self.mesh.disct_x, self.mesh.disct_z, \
                        self.mesh.surfpts, self.mesh.model) 
        return mesh
        
        
    def discretize(self):
        '''
        Discretize the fault and make the addresses of the mesh 
        '''
        p = self.mesh.xfault.flatten(order='F')
        q = self.mesh.yfault.flatten(order='F')
        r = self.mesh.zfault.flatten(order='F')
        
        rowp = self.mesh.xfault.shape[0]
        colp = self.mesh.xfault.shape[1]
        numpatch = 2*(rowp-1)*(colp-1)
        
        trired = np.empty(0, dtype = int, order = 'F')
        for i in range(1, colp):
            for j in range(1, rowp):
                trired = np.append(trired, [(i-1)*rowp+j-1, (i-1)*rowp+j, i*rowp+j])
                trired = np.append(trired, [(i-1)*rowp+j-1, i*rowp+j-1, i*rowp+j])
        trired = trired.reshape((numpatch, 3), order='C')
                
        mesh = self.NT1(trired, p, q, r, self.mesh.xfault, self.mesh.yfault, \
                        self.mesh.zfault, self.mesh.disct_x, self.mesh.disct_z, \
                        self.mesh.surfpts, self.mesh.model)
        
        return mesh
        
    # %%    
def Gorkhamesh(mesh, NT1):
    '''
    Create the non-planar fault mesh
    '''
    current = makemeshclass(mesh, NT1)
    current.initialize()
    
    # a. curved fault surface 
    current = makemeshclass(mesh, NT1)
    mesh = current.strike_surface()
    
    # b. fault down-dip curvature 
    current = makemeshclass(mesh, NT1)
    mesh = current.downdip_surface()
    
    # c. fault bottom along-strike curvature 
    current = makemeshclass(mesh, NT1)
    mesh = current.strike_bottom()
    
    # now discretize and name the index 
    current = makemeshclass(mesh, NT1)
    mesh = current.discretize()
    
    return mesh
    
    
    