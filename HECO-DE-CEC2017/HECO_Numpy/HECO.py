#from cec2017_np import CFunction
from cec2017_cy import CFunction
import numpy as np
import random
#import operator
from numpy import sin, cos, pi, e, exp, sqrt, log, tan
#from numba import jit
#@jit

def rand_normal(mu, sigma):
    uniform = random.random()
    z = sqrt(-2.0 * log(uniform)) * sin(2.0 * pi * uniform) 
    z = mu + sigma * z
    return z

def rand_cauchy(mu, gamma):
    uniform = random.random()
    z = mu + gamma * tan(pi * (uniform - 0.5))
    return z

def poissonRandom(expectedValue):
    n = 0 
    limit = exp(-expectedValue)
    x = random.random()
    while (x > limit) :
        n += 1
        x *= random.random()
    return n

class EECO:
    params = {'H': 6,
              'num_stg': 4,
              'FES_MAX' : 20000
            }
    def __init__(self, benchID, D):
        LU = CFunction().get_LU_Bound(benchID, D)
        self.lb, self.ub = LU[0], LU[1]
        mat = CFunction().load_mat(benchID, D)
        self.o, self.M, self.M1, self.M2 = (mat[_] for _ in range(4))
        
    def Init_P(self, NP, benchID, D):
        P_X = np.random.uniform(low = self.lb[0], high = self.ub[0], size = (NP, D + 7)) #init solutions
        for i in range(NP):
            CFunction().evaluate(benchID, D, P_X[i, :], self.o, self.M, self.M1, self.M2)
        return P_X
    
    def Init_Q(self, P, NP, Lambda, benchID, stg):
        sel_idx = np.random.choice(int(NP / 5), Lambda, replace = False) + int(4 * NP / 5)
        Q_X = P[sel_idx, :]
        return Q_X, sel_idx
    
    def Choose_Strategy(self, ql):
        stg = 0
        wheel = random.random()
        if wheel <= sum(ql[:1]):
            stg = 0
        elif wheel <sum(ql[:2]) and wheel > sum(ql[:1]):
            stg = 1
        elif wheel <sum(ql[:3]) and wheel > sum(ql[:2]):
            stg = 2
        elif wheel <sum(ql[:4]) and wheel > sum(ql[:3]):
            stg = 3
        return stg
    
    def mutation_1(self, P, NP, n_ctr, D):
        r_1 = poissonRandom(3)
        x = random.randint(NP / 5, NP - 1)
        x_1 = np.random.choice(int(NP / 5), n_ctr, replace = False)
        P1_ctr = np.sum(P[x_1, :], axis = 0) / n_ctr 
        gold = (1 + 5 ** 0.5) / 2
#        Y = 0.5 * (P1_ctr[:] + P[x, :]) + 0.5 * r_1 * (P1_ctr[:] - P[x, :])
        Y = (gold - 1) * P1_ctr[:] + (2 - gold) * P[x, :] + 0.5 * r_1 * (P1_ctr[:] - P[x, :])
        for i in range(D):
            if Y[i] < self.lb[i]:
                Y[i] = min(self.ub[i], 2 * self.lb[i] - Y[i]) 
            elif Y[i] > self.ub[i]:
                Y[i] = max(self.lb[i], 2 * self.ub[i] - Y[i])
        return Y
    
    def mutation_2(self, P, NP, n_ctr, D):
        r_1 = poissonRandom(3)
        x = random.randint(NP / 5, NP - 1)
        x_1 = np.random.choice(int(NP / 5), n_ctr, replace = False)
        P1_ctr = np.sum(P[x_1, :], axis = 0) / n_ctr 
        x_2 = np.random.choice(int(NP / 5), n_ctr, replace = False)
        P2_ctr = np.sum(P[int(NP / 5) + x_2, :], axis = 0) / n_ctr  
        gold = (1 + 5 ** 0.5) / 2
#        Y = 0.5 * (P1_ctr[:] + P[x, :]) + 0.5 * r_1 * (P2_ctr[:] - P[x, :])
#        Y = (gold - 1) * P1_ctr[:] + (2 - gold) * P[x, :] + 0.5 * r_1 * (P2_ctr[:] - P[x, :])
        Y = (gold - 1) * P1_ctr[:] + (2 - gold) * P[x, :] + 0.5 * r_1 * (P2_ctr[:] - P[x, :])
        for i in range(D):
            if Y[i] < self.lb[i]:
                Y[i] = min(self.ub[i], 2 * self.lb[i] - Y[i]) 
            elif Y[i] > self.ub[i]:
                Y[i] = max(self.lb[i], 2 * self.ub[i] - Y[i])
        return Y
    
    
    
    def crossover_1(self, P, C_X, NP, D):
        cr = 0.5 + 0.5 * random.random()
        jRand = random.randint(0, D - 1)
        for j in range(D):
            if jRand != j and random.random() <= cr:
                C_X[j] = P[random.randint(0, NP - 1), j]

                    
    def crossover_2(self, P, C_X, NP, D):
        cr = 0.5 + 0.5 * random.random()
        n = random.randint(0, D - 1)
        L = 0
        while random.uniform(0, 1) <= cr and L < D - 1:
            C_X[(n + L) % D] = P[random.randint(0, NP - 1), (n + L) % D]
            L += 1 
    
    def DE(self, P, NP, n_ctr, D, stg):
        if stg == 0:
            C_X = self.mutation_1(P, NP, n_ctr, D)
            self.crossover_1(P, C_X, NP, D)
        elif stg == 1:
            C_X = self.mutation_1(P, NP, n_ctr, D)
            self.crossover_2(P, C_X, NP, D)
        elif stg == 2:
            C_X = self.mutation_2(P, NP, n_ctr, D)
            self.crossover_1(P, C_X, NP, D)
        elif stg == 3:
            C_X = self.mutation_2(P, NP, n_ctr, D)
            self.crossover_2(P, C_X, NP, D)
        return C_X
    
    def Equ(self, P, NP, D):
        jugg = -1
        fea_idx = []
        f_max = np.amax(P[:, D])
        f_min = np.amin(P[:, D])
        v_max = np.amax(P[:, D + 1])
        v_min = np.amin(P[:, D + 1])
        for i in range(NP):
            if P[i, D + 1] == 0.0:
                jugg = 0
                fea_idx.append(i)
#                fea_f = P[i, D]
#                break
        
        if jugg == -1:
            P[:, D + 2] = (P[:, D + 1] - f_min) / (f_max - f_min + 10.0**-100) + (P[:, D + 1] - v_min) / (v_max - v_min + 10.0**-100)
#            P[:, D + 2] = P[:, D + 1]
        elif jugg == 0:
            fea_f = np.argmax(P[fea_idx, D])
            for i in range(NP):
                if P[i, D + 1] == 0.0:
                    P[i, D + 2] = P[i, D]
                elif P[i, D + 1] > 0.0:
                    P[i, D + 2] = (fea_f - f_min) / (f_max - f_min + 10.0**-100) + (P[i, D + 1] - v_min) / (v_max - v_min + 10.0**-100)
    
    def selection(self, P, QParent, QChild, NP, Lambda, D, idx, para, ql, stg, Gamma):
        QSum = np.concatenate((QParent, np.reshape(QChild[idx, :], (1, QChild.shape[1]))), axis = 0)
        f_max = np.amax(QSum[:, D])
        f_min = np.amin(QSum[:, D])
#        v_max = np.amax(QSum[:, D + 1])
#        v_min = np.amin(QSum[:, D + 1])
        self.Equ(QSum, Lambda, D)
        eq_max = np.amax(QSum[:, D + 2])
        eq_min = np.amin(QSum[:, D + 2])  
        w_t = para
        w_i = (idx + 1)/ Lambda
#        w1 = w_t**(20 * (idx + 1)) 
#        w2 = (1 - w_i) * (1 - w_t)**(5 * w_i)
#        w3 = w_i * w_t**(5 * w_i)
        w1 = w_i * w_t**(10 * w_i)
        w2 = (1 - w_i) * (1 - w_t)**(10 * w_i)

        
        QSum[:, D + 3] = w1 * (QSum[:, D + 2] - eq_min) / (eq_max - eq_min + 10.0**-100) 
        QSum[:, D + 3] += w2 * (QSum[:, D] - f_min) / (f_max - f_min + 10.0**-100)
#        QSum[:, D + 3] += w3 * (QSum[:, D + 1] - v_min) / (v_max - v_min + 10.0**-100)
        bestIndex = self.findBest(P, NP, D)
        if QSum[idx, D + 3] > QSum[-1, D + 3]:
#        if QParent[idx, D + 0] > QChild[idx, D + 0]:
#            print('replace*********************:{}, {}'.format(QParent[idx, D + 0], QChild[idx, D + 0]))
            if QParent[idx, D] != P[bestIndex, D] or QParent[idx, D + 1] != P[bestIndex, D + 1]:
#                if QChild[idx, D:D + 1] not in P[:, D:D + 1]:
#                    print(QChild[idx, D:D+2])
                QParent[idx, :] = QChild[idx, :]
            ql[stg] = ql[stg] + Gamma * (1 - ql[stg])
            for k in range(len(ql)):
                if k != stg:
                    ql[k] = ql[k] - ql[k] * Gamma
                    
        else:
            ql[stg] = ql[stg] - Gamma * ql[stg]
            for k in range(len(ql)):
                if k != stg:
                    ql[k] = ql[k] + ql[k] * Gamma
        ql_sum = 0
        ql_epsilon = 0.05 / len(ql)
        for k in range(len(ql)):
            ql_sum += ql_epsilon + ql[k]
        for k in range(len(ql)):
            ql[k] = (ql[k] + ql_epsilon) / ql_sum
    
    def findBest(self, P, NP, D):
        bestIndex = 0
        for i in range(NP):
            if P[bestIndex, D + 1] > P[i, D + 1]:
                bestIndex = i
            elif P[bestIndex, D + 1] == P[i, D + 1] and P[bestIndex, D] > P[i, D]:
                bestIndex = i
        return bestIndex
    
    def optimize(self, benchID, D):
        NP = 12 * D
        Lambda = 10
        n_ctr = 3
        Gamma = 1/3
        FES = NP
        gen_count = 1
        FES_MAX = 20000 * D
        P = self.Init_P(NP, benchID, D) #init population P
        ql = np.array([0.25, 0.25, 0.25, 0.25])
        QChild = np.zeros((Lambda, D + 7))
#        while FES < NP + 10 * Lambda + 1:
        while FES < FES_MAX:
            para = FES / FES_MAX
            stg = self.Choose_Strategy(ql)
            self.Equ(P, NP, D)
            if gen_count % 10 == 1:
                if para < 0.5:
                    P = P[np.argsort(P[:, D + 0])]
                else:
                    P = P[np.argsort(P[:, D + 2])]
#                P = P[np.argsort(P[:, D + 2])]
            #initialize sub-population Q
            Tuple_Q = self.Init_Q(P, NP, Lambda, benchID, stg)
            QParent = Tuple_Q[0]
            sel_idx = Tuple_Q[1]
            
            QChild[:] = QParent[:]
#            print('QParent1:{}'.format(QParent[:, D]))
#            print('QChild1:{}'.format(QChild[:, D]))
            
            for idx in range(Lambda):
                QChild[idx, :] = self.DE(P, NP, n_ctr, D, stg)
                CFunction().evaluate(benchID, D, QChild[idx, :], self.o, self.M, self.M1, self.M2) 
                self.selection(P, QParent, QChild, NP, Lambda, D, idx, para, ql, stg, Gamma)
#            print('QParent2:{}'.format(QParent[:, D]))
#            print('QChild2:{}'.format(QChild[:, D]))
            P[sel_idx, :] = QParent[:, :]
#            print(P.shape[0], P.shape[1], QParent.shape[0], QParent.shape[1])
            FES += Lambda
            gen_count += 1
            bestIndex = self.findBest(P, NP, D)
            print(FES, P[bestIndex, D], P[bestIndex, D + 1], P[1, D], P[1, D + 1])
        
        
        
        
        
        
        
        
        
        
        