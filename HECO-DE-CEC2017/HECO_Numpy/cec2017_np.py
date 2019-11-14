# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:14:29 2019

@author: burningxt
"""

import scipy.io as sio
import numpy as np
from numpy import sin, cos, pi, e, exp, sqrt
from numba import jit
@jit

def sgn(v): 
      if v < 0:
          return -1
      if v > 0:
          return 1
      return 0
  
class CFunction(object):
    def load_mat(self, benchID, D): #load .mat files for python
        M_D = {10: 'M_10', 30: 'M_30', 50: 'M_50', 100: 'M_100'}
        M1_D = {10: 'M1_10', 30: 'M1_30', 50: 'M1_50', 100: 'M1_100'}
        M2_D = {10: 'M2_10', 30: 'M2_30', 50: 'M2_50', 100: 'M2_100'}
        o = np.zeros((1, D))
        M, M1, M2 = (np.zeros((D, D)) for _ in range(3))
        if benchID in [1, 3, 4, 6, 7, 8, 9]:
            mat_contents = sio.loadmat('Function{}.mat'.format(benchID))
            o = mat_contents['o']
        elif benchID == 2:
            mat_contents = sio.loadmat('Function2.mat')
            o = mat_contents['o']
            M = mat_contents[M_D[D]]
        elif benchID == 5:
            mat_contents = sio.loadmat('Function5.mat')
            o = mat_contents['o']
            M1 = mat_contents[M1_D[D]]
            M2 = mat_contents[M2_D[D]]
        elif benchID in range(10, 21):
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o = mat_contents['o']
        elif benchID in range(21, 29):
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            M = mat_contents[M_D[D]]
        return o, M, M1, M2
    
    def benchmark(self, benchID, D, x, o, M, M1, M2):
        z, y, w, absZ = (np.zeros(D) for _ in range(4))
        g, h = (np.zeros(10, dtype = np.float64) for _ in range(2))
        f, f0, f1, g0, g1, g2, h0, h1, h2, h3, h4, h5 = (0.0 for _ in range(12))
        len_g, len_h = 0, 0
        if benchID == 1:   #benchmark functions 1-29
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += z[i]**2 - 5000.0 * cos(0.1 * pi * z[i]) - 4000.0
            g[0] = g0   
            len_g = 1
            
        elif benchID == 2:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    y[i] += z[j] * M[i, j]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += y[i]**2 - 5000.0 * cos(0.1 * pi * y[i]) - 4000.0
            g[0] = g0
            len_g = 1
        
        elif benchID == 3:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 = 0.0
                for j in range(i + 1):
                    f0 += z[j]
                f += f0**2
            for i in range(D):
                g0 += z[i]**2 - 5000.0 * cos(0.1 * pi * z[i]) - 4000.0
            for i in range(D):
                h0 += z[i] * sin(0.1 * pi * z[i])
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
        
        elif benchID == 4:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
            for i in range(D):
                g0 += z[i] * sin(2.0 * z[i])
            for i in range(D):
                g1 += z[i] * sin(z[i])
            g[0] = -g0 
            g[1] = g1
            len_g = 2
            
        elif benchID == 5:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += 100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
            for i in range(D):
                for j in range(D):
                    y[i] += z[j] * M1[i, j]
                    w[i] += z[j] * M2[i, j]
            for i in range(D):
                g0 += y[i]**2 - 50.0 * cos(2 * pi * y[i]) - 40.0
                g1 += w[i]**2 - 50.0 * cos(2 * pi * w[i]) - 40.0
            g[0] = g0
            g[1] = g1
            len_g = 2
                
        elif benchID == 6:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
            for i in range(D):
                h0 += -z[i] * sin(z[i])
            for i in range(D):
                h1 += z[i] * sin(pi * z[i])
            for i in range(D):
                h2 += - z[i] * cos(z[i])
            for i in range(D):
                h3 += z[i] * cos(pi * z[i])
            for i in range(D):
                h4 += z[i] * sin(2.0 * sqrt(abs(z[i])))
            for i in range(D):
                h5 += - z[i] * sin(2.0 * sqrt(abs(z[i])))
            h[0] = h0
            h[1] = h1
            h[2] = h2
            h[3] = h3
            h[4] = h4
            h[4] = h5
            len_h = 6
        
        elif benchID == 7:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i] * sin(z[i])
            for i in range(D):
                h0 += z[i] - 100.0 * cos(0.5 * z[i]) + 100.0
            h1 = -h0
            h[0] = h0
            h[1] = h1
            len_h = 2
            
        elif benchID == 8:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D])
            for i in range(int(D / 2.0)):
                y[i] = z[2 * i]
                w[i] = z[2 * i + 1]
            for i in range(int(D / 2.0)):
                temp0 = 0
                for j in range(i + 1):
                    temp0 += y[j]
                h0 += temp0**2
            for i in range(int(D / 2.0)):
                temp1 = 0
                for j in range(i + 1):
                    temp1 += w[j]
                h1 += temp1**2
            h[0] = h0
            h[1] = h1
            len_h = 2
            
        elif benchID == 9:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D])
            for i in range(int(D / 2.0)):
                y[i] = z[2 * i]
                w[i] = z[2 * i + 1] 
            g0 = 1.0
            for i in range(int(D / 2.0)):
                g0 = g0 * w[i]
            for i in range(int(D / 2.0) - 1.0):
                h0 += (y[i]**2 - y[i + 1])**2
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
                
        elif benchID == 10:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f = np.amax(z[:D]) 
            for i in range(D):
                temp0 = 0
                for j in range(i + 1):
                    temp0 += z[i]
                h0 += temp0**2
            for i in range(D - 1):
                h1 += (z[i] - z[i + 1])**2
            h[0] = h0
            h[1] = h1  
            len_h = 2
            
        elif benchID == 11:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]
            g0 = 1.0
            for i in range(D):
                g0 = g0 * z[i]
            for i in range(D - 1):
                h0 += (z[i] - z[i + 1])**2
            g[0] = g0
            h[0] = h0
            len_g = 1
            len_h = 1
        
        elif benchID == 12:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += z[i]**2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
            for i in range(D):
                g0 += abs(z[i])
            for i in range(D): 
                g1 += z[i]**2
            g[0] = 4.0 - g0
            g[1] = g1 - 4.0
            len_g = 2
         
        elif benchID == 13:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += (100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1.0)**2)
            for i in range(D):
                g0 += z[i]**2 - 10.0 * cos(2 * pi * z[i]) + 10.0
                g1 += z[i]
            g[0] = g0 - 100.0
            g[1] = g1 - 2.0 * D
            g[2] = 5.0 - g1
            len_g = 3
        
        elif benchID == 14:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f0 += z[i]**2
                f1 += cos(2.0 * pi * z[i])
            f = -20.0 * exp(- 0.2 * sqrt(1.0 / D * f0)) + 20.0 - exp(1.0 / D * f1) + e
            for i in range(1, D):
                g0 += z[i]**2
            for i in range(D):
                h0 += z[i]**2
            g[0] = g0 + 1.0 - abs(z[0])
            h[0] = h0 - 4.0
            len_g = 1
            len_h = 1
            
        elif benchID == 15:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                absZ[i] = abs(z[i])
            f = np.amax(absZ[:D])
            g0 = 0
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100.0 * D
            h[0] = cos(f) + sin(f)
            len_g = 1
            len_h = 1
            
        elif benchID == 16:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += abs(z[i])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100.0 *D
            h[0] = (cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1.0 + exp(1.0)
            len_g = 1
            len_h = 1
            
        elif benchID == 17:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            f1 = 1.0
            for i in range(D):
                f0 += z[i]**2
                f1 = f1 * cos(z[i] / sqrt(i + 1))
            f = 1.0 / 4000.0 * f0 + 1.0 - f1
            for i in range(D):
                g0 = 0.0
                for j in range(D): 
                    if(j == i):
                        g0 = f0 - z[j]**2
                g1 += sgn(abs(z[i]) - g0 - 1.0)
            h0 = f0
            g[0] = 1.0 - g1
            h[0] = h0 - 4.0 * D
            len_g = 1
            len_h = 1
         
        elif benchID == 18:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                if abs(y[i]) < 0.5:
                    z[i] = y[i]
                else:
                    z[i] = 0.5 * round(2.0 * y[i])
            for i in range(D):
                f += z[i]**2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
            for i in range(D):
                g0 += abs(y[i])
            for i in range(D):
                g1 += y[i]**2
            h1 = 1.0
            for i in range(D - 1):
                h0 += 100.0 * (y[i]**2 - y[i + 1])**2
                h1 = h1 * (sin(y[i] - 1.0))**2 * pi
            g[0] = 1.0 - g0
            g[1] = g1 - 100.0 * D
            h[0] = h0 + h1
            len_g = 2
            len_h = 1
         
        elif benchID == 19:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                f += (abs(z[i]))**0.5 + 2.0 * sin(z[i]**3)
            for i in range(D - 1):
                g0 += -10.0 * exp(- 0.2 * sqrt(z[i]**2 + z[i + 1]**2))
            for i in range(D):
                g1 += (sin(2.0 * z[i]))**2
            g[0] = g0 + (D - 1) * 10.0 / exp(-5.0)
            g[1] = g1 - 0.5 * D
            len_g = 2
            
        elif benchID == 20:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D - 1):
                f += 0.5 + ((sin(sqrt(z[i]**2 + z[i + 1]**2)))**2 - 0.5) / (1 + 0.001 * sqrt(z[i]**2 + z[i+1]**2))**2
            f += 0.5 + ((sin(sqrt(z[D - 1]**2 + z[0]**2)))**2 - 0.5) / (1 + 0.001 * sqrt(z[D - 1]**2 + z[0]**2))**2
            for i in range(D):
                g0 += z[i]
            g[0] = (cos(g0))**2 - 0.25 * cos(g0) - 0.125
            g[1] = exp(cos(g0)) - exp(0.25)
            len_g = 2
#            
        elif benchID == 21:
            for i in range(D):
                z[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += z[i] **2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
            for i in range(D):
                g0 += abs(z[i])
                g1 += z[i]**2
            g[0] = 4.0 - g0
            g[1] = g1 - 4.0
            len_g = 2
#            
        elif benchID == 22:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D - 1):
                f += 100.0 * (z[i]**2 - z[i + 1])**2 + (z[i] - 1)**2
            for i in range(D):
                g0 += z[i]**2 - 10.0 * cos(2.0 * pi * z[i]) + 10.0
                g1 += z[i]
            g[0] = g0 - 100.0
            g[1] = g1 - 2.0 * D
            g[2] = 5.0 - g1
            len_g = 3
                
        elif benchID== 23:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f0 += z[i]**2
                f1 += cos(2 * pi * z[i])
            f += -20.0 * exp(-0.2 * sqrt(1.0 / D * f0)) + 20.0 - exp(1.0 / D * f1) + e
            for i in range(1, D):
                g0 += z[i]**2
            for i in range(D):
                h0 += z[i]**2
            g[0] = g0 + 1.0 - abs(z[0])
            h[0] = h0 - 4.0
            len_g = 1
            len_h = 1
        
        elif benchID == 24:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                absZ[i] = abs(z[i])
            f = np.amax(absZ[:D])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100.0 * D
            h[0] = cos(f) + sin(f)
            len_g = 1
            len_h = 1
            
        if benchID == 25:  
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += abs(z[i])
            for i in range(D):
                g0 += z[i]**2
            g[0] = g0 - 100.0 * D
            h[0] = (cos(f) + sin(f))**2 - exp(cos(f) + sin(f)) - 1.0 + exp(1)
            len_g = 1
            len_h = 1
        
        elif benchID == 26:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            f1 = 1.0
            for i in range(D):
                f0 += z[i]**2
                f1 = f1 * cos(z[i] / sqrt(i + 1))
            f = 1.0 / 4000.0 * f0 + 1.0 - f1
            for i in range(D):
                g0 = 0.0
                for j in range(D): 
                    if(j == i):
                        g0 = f0 - z[j]**2
                g1 += sgn(abs(z[i]) - g0 - 1)
            h0 = f0
            g[0] = 1.0 - g1
            h[0] = h0 - 4.0 * D
            len_g = 1
            len_h = 1
            
        elif benchID == 27:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            if abs(z[i]) < 0.5:
                z[i] = y[i]
            else:
                z[i] = 0.5 * round(2 * y[i])
            for i in range(D):
                f += z[i]**2 - 10 * cos(2 * pi * z[i]) + 10
            for i in range(D):
                g0 += abs(y[i])
            for i in range(D):
                g1 += y[i]**2
            for i in range(D - 1):
                h0 += 100.0 * (y[i]**2 - y[i + 1])**2
                h1 = h1 * (sin(y[i] - 1))**2 * pi
            g[0] = 1.0 - g0
            g[1] = g1 - 100.0 * D
            h[0] = h0 + h1
            len_g = 2
            len_h = 1
            
        elif benchID == 28:
            for i in range(D):
                y[i] = x[i] - o[0, i]
            for i in range(D):
                for j in range(D):
                    z[i] += y[j] * M[i, j]
            for i in range(D):
                f += (abs(z[i]))**0.5 + 2 * sin(z[i]**3)
            for i in range(D - 1):
                g0 += -10 * exp(- 0.2 * sqrt(z[i]**2 + z[i + 1]**2))
            for i in range(D):
                g1 += (sin(2.0 * z[i]))**2
            g[0] = g0 + (D - 1.0) * 10.0 / exp(-5)
            g[1] = g1 - 0.5 * D
            len_g = 2
            
            
        #calculate c1, c2, c3
        c1, c2, c3, v = (0.0 for _ in range(4))
        if g != []:
            for i in range(len_g):
                if max(0.0, g[i]) > 1.0:
                    c1 += 1.0
                if max(0.0, g[i]) > 0.01 and max(0.0, g[i]) < 1.0:
                    c2 += 1.0
                if max(0.0, g[i]) > 0.0001 and max(0.0, g[i]) < 0.01:
                    c3 += 1.0
            for i in range(len_g):
                v += max(0.0, g[i])
        if h != []:
            for i in range(len_h):
                if max(0.0, abs(h[i]) - 0.0001) > 1.0:
                    c1 += 1.0
                if max(0.0, abs(h[i]) - 0.0001) > 0.01 and max(0.0, abs(h[i]) - 0.0001) < 1.0:
                    c2 += 1.0
                if max(0.0, abs(h[i]) - 0.0001) > 0.0001 and max(0.0, abs(h[i]) - 0.0001) < 0.01:
                    c3 += 1.0
            for i in range(len_h):
                v += max(0.0, abs(h[i]) - 0.0001)
        v = v / (len_g + len_h)
        
        return f, v, c1, c2, c3 #objevtive value and violation degree
            
    def get_LU_Bound(self, benchID, D): #get lower and upper bound of solutions
        if benchID in range(1, 4) or [8] or range(10, 19) or range(20, 28):
            lb = np.full(D, -100.0)
        elif benchID in [4, 5, 9]:
            lb = np.full(D, -10.0)
        elif benchID in [6]:
            lb = np.full(D, -20.0)
        elif benchID in [7, 19, 28]:
            lb = np.full(D, -50.0)
        if benchID in range(1, 4) or [8] or range(10, 19) or range(20, 28):
            ub = np.full(D, 100.0)
        elif benchID in [4, 5, 9]:
            ub = np.full(D, 10.0)
        elif benchID in [6]:
            ub = np.full(D, 20.0)
        elif benchID in [7, 19, 28]:
            ub = np.full(D, 50.0) 
        return lb, ub
    
    def evaluate(self, benchID, D, X, o, M, M1, M2):
        results = self.benchmark(benchID, D, X[:D], o, M, M1, M2)
        
        X[D] = results[0]
        X[D + 1] = results[1]
        X[D + 2] = 0.0
        X[D + 3] = results[2]
        X[D + 4] = results[3]
        X[D + 5] = results[4]
    



        
           
            
            
            
            
            
            
            
            
            
            
            
            