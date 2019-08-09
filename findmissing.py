#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:36:27 2019
Finds the missing samples in the present stage and copiess them

@author: duttar
"""
import numpy as np

lineList = np.array([line.rstrip('\n') for line in open('allind.txt')])

present = np.zeros((2000, 1))
for i in range(lineList.shape[0]):
    C = np.int(lineList[i].split('.')[0].split('e')[2]) 
    present[C-1] = 1

present = present.flatten('F')
missing = np.array(np.where(present == 0))
missing = missing.flatten('F')

vartoprint = np.str()
for i in range(missing.shape[0]):
    if i == 0:
        vartoprint = vartoprint + np.str(missing[i]+1)
    else:
        vartoprint = vartoprint + ',' + np.str(missing[i]+1)

vartorun = '#SBATCH --array=' + vartoprint
endnum = np.str(missing[-1] +1)

file1 = open("vartorun.txt","w") 
file1.write(vartorun) 
file1.close() 

file2 = open("endnum.txt","w") 
file2.write(endnum) 
file2.close() 

