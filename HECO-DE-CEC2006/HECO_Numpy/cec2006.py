# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:11:46 2018

@author: burningxt
"""

import scipy.io as sio
import math
from math import sin, cos, pi, e, exp, sqrt
import numpy as np


def sgn(v): 
      if v < 0:
          return -1
      elif v > 0.0:
          return 1
      else:
          return 0
      
def np_max(z):
        max_value = z[0]
        bound = z.shape[0]
        for i in range(bound):
            if max_value < z[i]:
                max_value = z[i]
        return max_value
    
class CFunction(object):
    def load_mat(self, benchID, D): #load .mat files for python
        o = np.zeros((1, D), dtype = np.float64)
        M = np.zeros((D, D), dtype = np.float64)
        M1 = np.zeros((D, D), dtype = np.float64)
        M2 = np.zeros((D, D), dtype = np.float64)
        M_D = {10: 'M_10', 30: 'M_30', 50: 'M_50', 100: 'M_100'}
        M1_D = {10: 'M1_10', 30: 'M1_30', 50: 'M1_50', 100: 'M1_100'}
        M2_D = {10: 'M2_10', 30: 'M2_30', 50: 'M2_50', 100: 'M2_100'}
        if benchID in [25]:
            mat_contents = sio.loadmat('Function7.mat'.format(benchID))
            o = mat_contents['o']
        if benchID in [26]:
            mat_contents = sio.loadmat('ShiftAndRotation.mat')
            o = mat_contents['o']
#        if benchID in [1, 3, 4, 6, 7, 8, 9]:
#            mat_contents = sio.loadmat('Function{}.mat'.format(benchID))
#            o = mat_contents['o']
#        elif benchID == 2:
#            mat_contents = sio.loadmat('Function2.mat')
#            o = mat_contents['o']
#            M = mat_contents[M_D[D]]
#        elif benchID == 5:
#            mat_contents = sio.loadmat('Function5.mat')
#            o = mat_contents['o']
#            M1 = mat_contents[M1_D[D]]
#            M2 = mat_contents[M2_D[D]]
#        elif benchID in range(10, 21):
#            mat_contents = sio.loadmat('ShiftAndRotation.mat')
#            o = mat_contents['o']
#        elif benchID in range(21, 29):
#            mat_contents = sio.loadmat('ShiftAndRotation.mat')
#            M = mat_contents[M_D[D]]
        return o, M, M1, M2
    
    def benchmark(self, benchID, D, x, o, M, M1, M2):
        f = 0.0
        g = []
        h = []
        z = np.zeros(D, dtype = np.float64)
        y = np.zeros(D, dtype = np.float64)
        w = np.zeros(D, dtype = np.float64)
        absZ = np.zeros(D, dtype = np.float64)
        if benchID == 25:
            h0, h1 = 0.0, 0.0
            for i in range(D):
                z[i] = x[i] - o[0][i]
            for i in range(D):
                f += z[i] * sin(z[i])
            for i in range(D):
                h0 += z[i] - 100.0 * cos(0.5 * z[i]) + 100.0
            h1 = -h0
            h.append(h0)
            h.append(h1)
        elif benchID == 26:
            h0 = 0.0
            for i in range(D):
                z[i] = x[i] - o[0][i]
            for i in range(D):
                f += z[i]
            g0 = 1.0
            for i in range(D):
                g0 = g0 * z[i]
            for i in range(D - 1):
                h0 += (z[i] - z[i + 1])**2
            g.append(g0)
            h.append(h0)
        if benchID == 1:
            f = 5 * (x[0] + x[1] + x[2] + x[3]) - 5 * (x[1]**2 + x[2]**2 + x[3]**2 + x[0]**2) - (x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12])
            f = f - (-15.0000000000)
            g.append(2 * x[0] + 2 * x[1] + x[9] + x[10] - 10)
            g.append(2 * x[0] + 2 * x[2] + x[9] + x[11] - 10)
            g.append(-8 * x[0] + x[9])
            g.append(-8 * x[1] + x[10])
            g.append(-8 * x[2] + x[11])
            g.append(-2 * x[3] - x[4] + x[9])
            g.append(-2 * x[5] - x[6] + x[10])
            g.append(-2 * x[7] - x[8] + x[11])
        elif benchID == 2:
            f1 = 0.0
            f2 = 1.0
            f3 = 0.0
            k1 = 1.0
            k2 = 0.0
            for i in range(20):
                f1 += (math.cos(x[i]))**4
            for i in range(20):
                f2 = f2 * (math.cos(x[i]))**2
            for i in range(20):
                f3 += (i + 1) * x[i]**2 
            f = -math.fabs((f1 - 2 * f2) / (10**-20 + math.sqrt(f3)))
            f = f - (-0.8036191042)
            for i in range(20):
                k1 = k1 * x[i] 
                g.append(0.75 - k1)
            for i in range(20):
                k2 += x[i]
                g.append(k2 - 7.5 * 20)
        elif benchID == 3:
            f1 = 1.0
            k1 = 0.0
            for i in range(10):
                f1 = f1 * x[i]
            f = - (math.sqrt(10.0))**10 * f1
            f = f - (-1.0005001000)
            for i in range(10):
                k1 += x[i]**2
            h.append(k1 - 1)
        elif benchID == 4:
            f = 5.3578547 * x[2]**2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
            f = f - (-30665.5386717834)
            g.append(85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92)
            g.append(-85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4])
            g.append(80.51249 + 0.0071317 * x[1] * x[4] + 0.002995 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110)
            g.append(-80.51249 - 0.0071317 * x[1] * x[4] - 0.002955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90) 
            g.append(9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] -25)
            g.append(-9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20)
        elif benchID == 5:
            f = 3 * x[0] + 0.000001 * x[0]**3 + 2 * x[1] + (0.000002 / 3) *  x[1]**3
            f = f - (5126.4967140071)
            g.append(- x[3] + x[2] - 0.55)
            g.append(- x[2] + x[3] - 0.55)
            h.append(1000 * math.sin(- x[2] - 0.25) + 1000 * math.sin(- x[3] - 0.25) + 894.8 - x[0])
            h.append(1000 * math.sin(x[2] - 0.25) + 1000 * math.sin(x[2] - x[3] - 0.25) + 894.8 - x[1])
            h.append(1000 * math.sin(x[3] - 0.25) + 1000 * math.sin(x[3] - x[2] - 0.25) + 1294.8)
        elif benchID == 6:
            f = (x[0] - 10)*(x[0] - 10)*(x[0] - 10) + (x[1] - 20)*(x[1] - 20)*(x[1] - 20)
            f = f - (-6961.8138755802)
            g.append(-(x[0] - 5)*(x[0] - 5) - (x[1] - 5)*(x[1] - 5) + 100)
            g.append((x[0] - 6)*(x[0] - 6) + (x[1] - 5)*(x[1] - 5) - 82.81)
        elif benchID == 7:
            f = x[0]**2 + x[1]**2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10)**2 + 4 * (x[3] - 5)**2 + (x[4] - 3)**2 + 2 * (x[5] - 1)**2 + 5 * x[6]**2 + 7 * (x[7] - 11)**2 + 2 * (x[8] - 10)**2 + (x[9] - 7)**2 + 45
            f = f - 24.3062090681
            g.append(- 105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7])
            g.append(10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7])
            g.append(- 8 * x[0] + 2 * x[1] + 5 * x[8] -2 * x[9] - 12)
            g.append(3 * (x[0]- 2)**2 + 4 * (x[1] - 3)**2 + 2 * x[2]**2 - 7 * x[3] - 120)
            g.append(5 * x[0]**2 + 8 * x[1] + (x[2] - 6)**2 - 2 * x[3] - 40)
            g.append(x[0]**2 + 2 * (x[1] - 2)**2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5])
            g.append(0.5 * (x[0] - 8)**2 + 2 * (x[1] - 4)**2 + 3 * x[4]**2 - x[5] - 30)
            g.append(- 3 * x[0] + 6 * x[1] + 12 * (x[8] - 8)**2 - 7 * x[9])
        elif benchID == 8:
            if x[0]**3.0 * (x[0] + x[1]) != 0.0:
                f = - (math.sin(2 * math.pi * x[0]))**3 * math.sin(2 * math.pi * x[1]) / (x[0]**3.0 * (x[0] + x[1]))
            else:
                f = - (math.sin(2 * math.pi * x[0]))**3 * math.sin(2 * math.pi * x[1]) / (x[0]**3.0 * (x[0] + x[1]) + 10.0**-100)
            f = f - (-0.0958250415)   
            g.append(x[0]**2 - x[1] + 1)
            g.append(1 - x[0] + (x[1] - 4)**2)
        elif benchID == 9:
            f = (x[0] - 10)**2 + 5 * (x[1] - 12)**2 + x[2]**4 + 3 * (x[3] - 11)**2 + 10 * x[4]**6 +7 * x[5]**2 + x[6]**4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6]
            f = f - 680.6300573745
            g.append(-127 + 2 * x[0]**2 + 3 * x[1]**4 + x[2] + 4 * x[3]**2 + 5 * x[4])
            g.append(-282 + 7 * x[0] + 3 * x[1] + 10 * x[2]**2 + x[3] - x[4])
            g.append(-196 + 23 * x[0] + x[1]**2 + 6 * x[5]**2 - 8 * x[6])
            g.append(4 * x[0]**2 + x[1] **2 - 3 * x[0]*x[1] + 2 * x[2]**2 + 5 * x[5] - 11 * x[6])
        elif benchID == 10:
            f = x[0] + x[1] + x[2]
            f = f - 7049.2480205286
            g.append(-1 + 0.0025 * (x[3] + x[5]))
            g.append(-1 + 0.0025 * (x[4] + x[6] - x[3]))
            g.append(-1 + 0.01 * (x[7] - x[4]))
            g.append(- x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333)
            g.append(- x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3])
            g.append(-x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4])
        elif benchID == 11:
            f = x[0]**2 + (x[1] - 1)**2
            f = f - 0.7499000000
            h.append(x[1] - x[0]**2)
        elif benchID == 12:
            f = - (100 - (x[0] - 5)**2 - (x[1] - 5)**2 - (x[2] - 5)**2) / 100
            f = f - (-1.0000000000)
            g.append((x[0] - 1)**2 + (x[1] - 1)**2 + (x[2] - 1)**2 - 0.0625)
            for p in range(1, 10):
                for q in range(1, 10):
                    for r in range(1, 10):
                        gt = (x[0] - p)**2 + (x[1] - q)**2 + (x[2] - r)**2 - 0.0625 
                        if(gt < g[0]):
                            g[0] = gt
        elif benchID == 13:
            f = math.exp(x[0] * x[1] * x[2] * x[3] * x[4])
            f = f - 0.0539415140
            h.append(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10)
            h.append(x[1] * x[2] - 5 * x[3] * x[4])
            h.append(x[0]**3 + x[1]**3 + 1)
        elif benchID == 14:
            c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179]
            f1 = 0
            f2 = 0
            for i in range(10):
                f1 += x[i]
            for i in range(10):
                f2 += x[i] * (c[i] + math.log(x[i] / f1))
            f = f2
            f = f - (-47.7648884595)
            h.append(x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] -2)
            h.append(x[3] + 2 * x[4] + x[5] + x[6] -1)
            h.append(x[2] + x[6] + x[7] + 2* x[8] + x[9] -1)
        elif benchID == 15:
            f = 1000 - x[0]**2 - 2 * x[1]**2 - x[2]**2 - x[0] * x[1] - x[0] * x[2]
            f = f - 961.7150222899
            h.append(x[0]**2 + x[1]**2 + x[2]**2 - 25)
            h.append(8 * x[0] + 14 * x[1] + 7 * x[2] - 56)
        elif benchID == 16:
            y1 = x[1] + x[2] + 41.6
            c1 = 0.024 * x[3] - 4.62
            y2 = 12.5 / c1 + 12
            c2 = 0.0003535 * x[0]**2 + 0.5311 * x[0] + 0.08705 * y2 * x[0]
            c3 = 0.052 * x[0] + 78 + 0.002377 * y2 * x[0]
            y3 = c2 / c3
            y4 = 19 * y3
            c4 = 0.04782 * ( x[0] - y3) + 0.1956 * (x[0] - y3)**2 / x[1] + 0.6376 * y4 + 1.594 * y3
            c5 = 100 * x[1]
            c6 = x[0] - y3 - y4
            c7 = 0.950 - c4 / c5
            y5 = c6 * c7
            y6 = x[0] - y5 - y4 - y3
            c8 = (y5 + y4) * 0.995
            y7 = c8 / y1
            y8 = c8 / 3798
            c9 = y7 - 0.0663 * y7 / y8 - 0.3153
            y9 = 96.82 / c9 + 0.321 * y1
            y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
            y11 = 1.71 * x[0] - 0.452 * y4 + 0.580 * y3
            c10 = 12.3 / 752.3 
            c11 = 1.75 * y2 * 0.995 * x[0]
            c12 = 0.995 * y10 + 1998
            y12 = c10 * x[0] + c11 / c12
            y13 = c12 - 1.75 * y2
            y14 = 3623 + 64.4 * x[1] + 58.4 * x[2] + 146312 / (y9 + x[4])
            c13 = 0.995 * y10 + 60.8 * x[1] + 48 * x[3] - 0.1121 * y14 - 5095
            y15 = y13 / c13
            y16 = 148000 - 331000 * y15 + 40 * y13 - 61 * y15 * y13
            c14 = 2324 * y10 - 28740000 * y2
            y17 = 14130000 - 1328 * y10 - 531 * y11 + c14 / c12
            c15 = y13 / y15 - y13 / 0.52
            c16 = 1.104 - 0.72 * y15
            c17 = y9 + x[4]
            
            f = 0.000117 * y14 + 0.1365 + 0.00002358 * y13 + 0.000001502 * y16 + 0.0321 * y12 + 0.004324 * y5 + 0.0001 * c15 / c16 + 37.48 * y2 / c12 - 0.0000005843 * y17 
            f = f - (-1.9051552586)
            g.append(0.28 / 0.72 * y5 - y4)
            g.append(x[2] - 1.5 * x[1])
            g.append(3496.0 * y2 / c12 - 21)
            g.append(110.6 + y1 - 62212.0 / c17)
            g.append(213.1 - y1)
            g.append(y1 - 405.23)
            g.append(17.505 - y2)
            g.append(y2 - 1053.6667)
            g.append(11.275 - y3)
            g.append(y3 - 35.03)
            g.append(214.228 - y4)
            g.append(y4 - 665.585)
            g.append(7.458 - y5)
            g.append(y5 - 584.463)
            g.append(0.961 - y6)
            g.append(y6 - 265.916)
            g.append(1.612 - y7)
            g.append(y7 - 7.046)
            g.append(0.146 - y8)
            g.append(y8 - 0.222)
            g.append(107.99 - y9)
            g.append(y9 - 273.366)
            g.append(922.693 - y10)
            g.append(y10 - 1286.105)
            g.append(926.832 - y11)
            g.append(y11 - 1444.046)
            g.append(18.766 - y12)
            g.append(y12 - 537.141)
            g.append(1072.163 - y13)
            g.append(y13 - 3247.039)
            g.append(8961.448 - y14)
            g.append(y14 - 26844.086)
            g.append(0.063 - y15)
            g.append(y15 - 0.386)
            g.append(71084.33 - y16)
            g.append(-140000 + y16)
            g.append(2802713 - y17)
            g.append(y17 - 12146108)
        elif benchID == 17:
            f1 = 0.0
            f2 = 0.0
            if x[0] >= 0.0 and x[0] < 300.0:
                f1 = 30 * x[0]
            elif x[0] >= 300.0 and x[0] < 400.0:
                f1 = 31 * x[0]
            if x[1] >= 0.0 and x[1] < 100.0:
                f2 = 28 * x[1]
            elif x[1] >= 100.0 and x[1] < 200.0:
                f2 = 29 * x[1]
            elif x[2] >= 200.0 and x[2] < 1000.0:
                f2 = 30 * x[1]
            f = f1 + f2
            f = f - 8853.533874806484
            h.append(- x[0] + 300 - x[2] * x[3] / 131.078 * cos(1.48477 - x[5]) + 0.90798 * x[2]**2 / 131.078 * cos(1.47588))
            h.append(- x[1] - x[2] * x[3] / 131.078 * cos(1.48477 + x[5]) + 0.90798 * x[3]**2 / 131.078 * cos(1.47588))
            h.append(- x[4] - x[2] * x[3] / 131.078 * sin(1.48477 + x[5]) + 0.90798 * x[3]**2 / 131.078 * sin(1.47588))
            h.append(200 - x[2] * x[3] / 131.078 * sin(1.48477 - x[5]) + 0.90798 * x[2]**2 / 131.078 * sin(1.47588))
        elif benchID == 18:
            f = - 0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])
            f = f - (-0.8660254038)
            g.append(x[2]**2 + x[3]**2 - 1)
            g.append(x[8]**2 - 1)
            g.append(x[4]**2 + x[5]**2 -1)
            g.append(x[0]**2 + (x[1] - x[8])**2 - 1)
            g.append((x[0] - x[4])**2 + (x[1] - x[5])**2 - 1)
            g.append((x[0] - x[6])**2 + (x[1] - x[7])**2 - 1)
            g.append((x[2] - x[4])**2 + (x[3] - x[5])**2 - 1)
            g.append((x[2] - x[6])**2 + (x[3] - x[7])**2 - 1)
            g.append(x[6]**2 + (x[7] - x[8])**2 - 1)
            g.append(x[1] * x[2] - x[0] * x[3])
            g.append(- x[2] * x[8])
            g.append(x[4] * x[8])
            g.append(x[5] * x[6] - x[4] * x[7])
        elif benchID == 19:
            a = [[-16, 2, 0, 1, 0], [0, -2, 0, 0.4, 2], [-3.5, 0, 2, 0, 0], [0, -2, 0, -4, -1], [0, -9, -2, 1, -2.8], [2, 0, -4, 0, 0], [-1, -1, -1, -1, -1], [-1, -2, -3, -2, -1], [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
            b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
            c = [[30, -20, -10, 32, -10], [-20, 39, -6, -31, 32], [-10, -6, 10, -6, -10], [32, -31, -6, 39, -20], [-10, 32, -10, -20, 30]]
            d = [4, 8, 10, 6, 2]
            e = [-15, -27, -36, -18, -12]
            f1 = 0
            f2 = 0
            f3 = 0
            for j in range(5):
                for i in range(5):
                    f1 += c[i][j] * x[10 + i] * x[10 + j]
            for i in range(5):
                f2 += d[i] * x[10 + i]**3
            for i in range(10):
                f3 += b[i] * x[i]
            f = f1 + 2 * f2 - f3
            f = f - 32.6555929502
            for j in range(5):
                g1 = 0
                g2 = 0
                for i in range(5):
                    g1 += c[i][j] * x[10 + i] 
                for i in range(10):
                    g2 += a[i][j] * x[i]
                g.append(-2 * g1 - 3 * d[j] * x[10 + j]**2 - e[j] + g2)
        elif benchID == 20:
            a = [0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09, 0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09]
            b = [44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.079]
            c = [123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
            d = [31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1]
            e = [0.1, 0.3, 0.4, 0.3, 0.6, 0.3]
            f1 = 0
            for i in range(24):
                f1 += a[i] * x[i]
            f = f1
            f = f - 0.2049794002
            for i in range(3):
                g1 = 0
                for j in range(24):
                    g1 += x[j]
                g.append((x[i] + x[i + 12]) / (g1 + e[i]))
            for j in range(3):
                g2 = 0
                for j in range(24):
                    g2 += x[j]
                g.append((x[i + 3] + x[i + 15]) / (g2 + e[i]))
            for i in range(12):
                h1 = 0
                h2 = 0
                for j in range(12, 24):
                    h1 += x[j] / b[j]
                for j in range(12):
                    h2 += x[j]/b[j]
                h.append(x[i + 12] / (b[i + 12] * h1) - c[i] * x[i] / (40 * b[i] * h2))
            h3 = 0
            h4 = 0
            h5 = 0
            for j in range(12, 24):
                h4 += x[j] / b[j]
            for j in range(12):
                h5 += x[j]/b[j]
            for j in range(24):
                h3 += x[j]
            h.append(h3 - 1)
            h.append(h4 + 0.7302 * 530 * 14.7 / 40 * h5 - 1.671)
        elif benchID == 21:
            f = x[0]
            f = f - 193.7245100700
            g.append(-x[0] + 35 * x[1]**0.6 + 35 * x[2]**0.6)
            h.append(-300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3])
            h.append(100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5)
            h.append(- x[4] + np.log(- x[3] + 900.0))
            h.append(- x[5] + np.log(x[3] + 300.0))
            h.append(- x[6] + np.log(- 2.0 * x[3] + 700.0))
        elif benchID == 22:
            f = x[0]
            f = f - 236.4309755040
            g.append(-x[0] + x[1]**0.6 + x[2]**0.6 + x[3]**0.6)
            h.append(x[4] - 100000 * x[7] + 10**7)
            h.append(x[5] + 100000 * x[7] - 100000 * x[8])
            h.append(x[6] + 100000 * x[8] - 5 * 10**7)
            h.append(x[4] + 100000 * x[9] -3.3 * 10**7)
            h.append(x[5] + 100000 * x[10] - 4.4 * 10**7)
            h.append(x[6] + 100000 * x[11] - 6.6 * 10**7)
            h.append(x[4] - 120 * x[1] * x[12])
            h.append(x[5] - 80 * x[2] * x[13])
            h.append(x[6] - 40 * x[3] * x[14])
            h.append(x[7] - x[10] + x[15])
            h.append(x[8] - x[11] + x[16])
            h.append(-x[17] + np.log(x[9] - 100.0 + 10.0**-20))
            h.append(-x[18] + np.log(-x[7] + 300.0))
            h.append(-x[19] + np.log(x[15] + 10.0**-20))
            h.append(-x[20] + np.log(-x[8] + 400.0))
            h.append(-x[21] + np.log(x[16]))
            h.append(-x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400)
            h.append(x[7] - x[8] - x[10] + x[13] * x[19] - x[13] * x[20] + 400)
            h.append(x[8] - x[12] - 4.60517 + x[14] * x[21] + 100)
        elif benchID == 23:
            f = -9 * x[4] -15 * x[7] + 6 * x[0] + 16 * x[1] + 10 * (x[5] + x[6])
            f = f - (-400.0551000000)
            g.append(x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4])
            g.append(x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7])
            h.append(x[0] + x[1] - x[2] - x[3])
            h.append(0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3]))
            h.append(x[2] + x[5] - x[4])
            h.append(x[3] + x[6] - x[7])
        elif benchID == 24:
            f = - x[0] - x[1]
            f = f - (-5.5080132716)
            g.append(-2 * x[0]**4 + 8 * x[0]**3 - 8 * x[0]**2 + x[1] - 2)
            g.append(-4 * x[0]**4 + 32 * x[0]**3 - 88 * x[0] ** 2 + 96 * x[0] + x[1] - 36)

        s1 = 0 
        s2 = 0
        s3 = 0
        v = 0.0
        if g != []:
            for i in range(len(g)):
                if max(0.0, g[i]) > 1:
                    s1 += 1
                if max(0.0, g[i]) > 0.01 and max(0.0, g[i]) < 1:
                    s2 += 1
                if max(0.0, g[i]) > 0.0001 and max(0.0, g[i]) < 0.01:
                    s3 += 1
            for i in range(len(g)):
                v += max(0.0, g[i])
        if h != []:
            for i in range(len(h)):
                if max(0.0, abs(h[i]) - 0.0001) > 1:
                    s1 += 1
                if max(0.0, abs(h[i]) - 0.0001) > 0.01 and max(0.0, abs(h[i]) - 0.0001) < 1:
                    s2 += 1
                if max(0.0, abs(h[i]) - 0.0001) > 0.0001 and max(0.0, abs(h[i]) - 0.0001) < 0.01:
                    s3 += 1
            for i in range(len(h)):
                v += max(0.0, abs(h[i]) - 0.0001)
        v = v / (len(g) + len(h))
        return [f, v, s1, s2, s3]
    def get_LB(self, benchID, D):
        lb = []
        if benchID == 1:
            for i in range(13):
                lb.append(0.0) 
        elif benchID == 2:
            for i in range(20):
                lb.append(10**-20) 
        elif benchID == 3:
            for i in range(10):
                lb.append(0.0)
        elif benchID == 4:
            lb.append(78)
            lb.append(33)
            for i in range (3):
                lb.append(27)
        elif benchID == 5:
            lb.append(0.0)
            lb.append(0.0)
            lb.append(-0.55)
            lb.append(-0.55)
        elif benchID == 6:
            lb.append(13.0)
            lb.append(0.0)
        elif benchID == 7:
            for i in range(10):
                lb.append(-10)
        elif benchID == 8:
            lb.append(0)
            lb.append(0)
        elif benchID == 9:
            for i in range(7):
                lb.append(-10)
        elif benchID == 10:
            lb.append(100.0)
            [lb.append(1000.0) for i in range(2)]
            [lb.append(10.0) for i in range(5)]
        elif benchID == 11:
            lb.append(-1)
            lb.append(-1)
        elif benchID == 12:
            for i in range(3):
                lb.append(0)
        elif benchID == 13:
            for i in range(2):
                lb.append(-2.3)
            for i in range(2, 5):
                lb.append(-3.2)
        elif benchID == 14:
            for i in range(10):
                lb.append(0)
        elif benchID == 15:
            for i in range(3):
                lb.append(0)
        elif benchID == 16:
            lb.append(704.4148)
            lb.append(68.6)
            lb.append(0.0)
            lb.append(193.0)
            lb.append(25.0)
        elif benchID == 17:
            lb.append(0.0)
            lb.append(0.0)
            lb.append(340.0)
            lb.append(340.0)
            lb.append(-1000.0)
            lb.append(0.0)
        elif benchID == 18:
            for i in range(8):
                lb.append(-10)
            lb.append(0)
        elif benchID == 19:
            for i in range(15):
                lb.append(0)
        elif benchID == 20:
            for i in range(24):
                lb.append(0)
        elif benchID == 21:
            for i in range(3):
                lb.append(0)
            lb.append(100)
            lb.append(6.3)
            lb.append(5.9)
            lb.append(4.5)
        elif benchID == 22:
            for i in range(7):
                lb.append(0)
            for i in range(7, 9):
                lb.append(100)
                lb.append(100.01)
            for i in range(10, 12):
                lb.append(100)
            for i in range(12, 15):
                lb.append(0)
            for i in range(15, 17):
                lb.append(0.01)
            for i in range(17, 22):
                lb.append(-4.7)
        elif benchID == 23:
            for i in range(8):
                lb.append(0)
            lb.append(0.01)
        elif benchID == 24:
            lb.append(0)
            lb.append(0)
        elif benchID == 25:
            for i in range(D):
                lb.append(-50.0)
        elif benchID == 26:
            for i in range(D):
                lb.append(-100.0)
        return lb
    def get_UB(self, benchID, D):
        ub = []
        if benchID == 1:
            for i in range(9):
                ub.append(1.0) 
            for i in range(9, 12):
                ub.append(100) 
            ub.append(1.0)
        elif benchID == 2:
            for i in range(20):
                ub.append(10.0)
        elif benchID == 3:
            [ub.append(10.0) for i in range(10)]
        elif benchID == 4:
            ub.append(102)
            for i in range(4):
                ub.append(45)
        elif benchID == 5:
            ub.append(1200.0)
            ub.append(1200.0)
            ub.append(0.55)
            ub.append(0.55)
        elif benchID == 6:
            ub.append(100.0)
            ub.append(100.0)
        elif benchID == 7:
            for i in range(10):
                ub.append(10)
        elif benchID == 8:
            ub.append(10)
            ub.append(10)
        elif benchID == 9:
            for i in range(7):
                ub.append(10)
        elif benchID == 10:
            [ub.append(10000.0) for i in range(3)]
            [ub.append(1000.0) for i in range(5)]
        elif benchID == 11:
            ub.append(1)
            ub.append(1)
        elif benchID == 12:
            for i in range(3):
                ub.append(10)
        elif benchID == 13:
            for i in range(2):
                ub.append(2.3)
            for i in range(2, 5):
                ub.append(3.2)
        elif benchID == 14:
            for i in range(10):
                ub.append(10)
        elif benchID == 15:
            for i in range(3):
                ub.append(10)
        elif benchID == 16:
            ub.append(906.3855)
            ub.append(288.88)
            ub.append(134.75)
            ub.append(287.0966)
            ub.append(84.1988)
        elif benchID == 17:
            ub.append(400)
            ub.append(1000)
            ub.append(420)
            ub.append(420)
            ub.append(1000)
            ub.append(0.5236)
        elif benchID == 18:
            for i in range(8):
                ub.append(10)
            ub.append(20)
        elif benchID == 19:
            for i in range(15):
                ub.append(10)
        elif benchID == 20:
            for i in range(24):
                ub.append(10)
        elif benchID == 21:
            ub.append(1000)
            ub.append(40)
            ub.append(40)
            ub.append(300)
            ub.append(6.7)
            ub.append(6.4)
            ub.append(6.25)
        elif benchID == 22:
            ub.append(20000)
            for i in range(1, 4):
                ub.append(10**6)
            for i in range(4,7):
                ub.append(4 * 10**7)
            ub.append(299.99)
            ub.append(399.99)
            ub.append(300)
            ub.append(400)
            ub.append(600)
            for i in range(12, 15):
                ub.append(500)
                ub.append(300)
                ub.append(400)
            for i in range(17, 22):
                ub.append(6.25)
        elif benchID == 23:
            ub.append(300)
            ub.append(300)
            ub.append(100)
            ub.append(200)
            ub.append(100)
            ub.append(300)
            ub.append(100)
            ub.append(200)
            ub.append(0.03)
        elif benchID == 24:
            ub.append(3)
            ub.append(4)
        elif benchID == 25:
            for i in range(D):
                ub.append(50.0)
        elif benchID == 26:
            for i in range(D):
                ub.append(100.0)
        return ub
    def getParam(self, benchID):
        paraNum = 0
        if benchID == 1:
            paraNum = 13
        elif benchID == 2:
            paraNum = 20
        elif benchID == 3:
            paraNum = 10
        elif benchID == 4:
            paraNum = 5
        elif benchID == 5:
            paraNum = 4
        elif benchID == 6:				
            paraNum = 2	
        elif benchID == 7:
            paraNum = 10
        elif benchID == 8:
            paraNum = 2
        elif benchID == 9:
            paraNum = 7
        elif benchID == 10:
            paraNum = 8
        elif benchID == 11:
            paraNum = 2
        elif benchID == 12:
            paraNum = 3
        elif benchID == 13:
            paraNum = 5
        elif benchID == 14:
            paraNum = 10
        elif benchID == 15:
            paraNum = 3
        elif benchID == 16:
            paraNum = 5
        elif benchID == 17:
            paraNum = 6
        elif benchID == 18:
            paraNum = 9
        elif benchID == 19:
            paraNum = 15
        elif benchID == 20:
            paraNum = 24
        elif benchID == 21:
            paraNum = 7
        elif benchID == 22:
            paraNum = 22	
        elif benchID == 23:
            paraNum = 9
        elif benchID == 24:
            paraNum = 2
        elif benchID == 25:
            paraNum = 10
        elif benchID == 26:
            paraNum = 10
        return paraNum
    
    def evaluate(self, benchID, D, X, o, M, M1, M2):
        results = self.benchmark(benchID, D, X[:D], o, M, M1, M2)
        X[D] = results[0]
        X[D + 1] = results[1]
        X[D + 2] = 0.0
        X[D + 3] = 0.0
        X[D + 4] = results[2]
        X[D + 5] = results[3]
        X[D + 6] = results[4]
    