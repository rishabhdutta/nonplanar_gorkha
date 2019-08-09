# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:50:21 2019

@author: duttar
"""
from Gorkhamakemesh import *
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from collections import namedtuple
import matplotlib.pyplot as plt
#runfile('Gorkhamakemesh.py', wdir='.')

NT1 = namedtuple('NT1', \
'trired p q r xfault yfault zfault disct_x disct_z surfpts model')

xdisct = 20; zdisct = 12
surfpts = np.array([[215.8972, 3.0950e3],[442.8739, 3.0097e3]])
model = np.array([-0.5185,   -0.2322 ,  16.3457 , -21.6003  ,  0.6674 , -0.1404 ,  -3.0565 ,   0.1159 , -12.6384 ,  -5.9212])
#model = np.array([1.0927  ,  0.0241 ,  5.6933 , -20.0000 ,   2.1345  ,  0.3529  ,  4.4643 ,   0.0336,  -12 , -2.8494  ,  0.0043])

mesh = NT1(None, None, None, None, None, None, None, \
            xdisct, zdisct, surfpts, model)
            
# get the geometry 
finalmesh = Gorkhamesh(mesh, NT1)          

# %%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


fig = plt.figure(figsize=plt.figaspect(0.5))

# Plot the surface.  The triangles in parameter space determine which x, y, z
# points are connected by an edge.
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(finalmesh.p, finalmesh.q, finalmesh.r, \
                triangles=finalmesh.trired, cmap=plt.cm.Spectral)
plt.axis('equal')
ax.set(xlim=(200, 500), ylim=(2900, 3400), zlim=(-25, -5))
plt.show()
#ax.set_zlim(-1, 1)
