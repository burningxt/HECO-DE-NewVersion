# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:03:05 2019

@author: burningxt
"""
import timeit
import numpy as np

cdef selection():
    cdef:
        double f_max
        double f_min
        double eq_max
        double eq_min
        double[:, :] QSum
    QSum = np.random.uniform(low = 0.0, high = 100.0, size = (180, 107))   
#    f_max = np.amax(QSum[:, 100])
#    f_min = np.amin(QSum[:, 100])
#    eq_max = np.amax(QSum[:, 100 + 2])
#    eq_min = np.amin(QSum[:, 100 + 2])

start = timeit.default_timer()
#cdef:
#    double[:, :] QSum
#QSum = np.random.uniform(low = 0.0, high = 100.0, size = (180, 107))       
for i in range(10000):
    selection()
stop = timeit.default_timer()
print('Time: ', stop - start) 