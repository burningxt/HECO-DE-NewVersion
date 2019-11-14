#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

#from cec2017_np import CFunction
from cec2017_cy import CFunction
import numpy as np
#cimport numpy as np
#np.import_array()
import random
import operator
from libc.math cimport sin, cos, pi, e, exp, sqrt, log, tan
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time
cimport cython


cdef double rand_normal(double mu, double sigma):
    cdef:
        double z, uniform
    uniform = random.random()
    z = sqrt(- 2.0 * log(uniform)) * sin(2.0 * pi * uniform) 
    z = mu + sigma * z
    return z

cdef double rand_cauchy(double mu, double gamma):
    cdef:
        double z, uniform
    uniform = random.random()
    z = mu + gamma * tan(pi * (uniform - 0.5))
    return z

#cdef int poissonRandom(int expectedValue):
#    cdef:
#        int  n
#        double limit, x 
#    n = 0 
#    limit = exp(-expectedValue)
#    x = <double>rand() / RAND_MAX
#    while (x > limit) :
#        n += 1
#        x *= <double>rand() / RAND_MAX
#    return n

cdef int rand_int(int r_min, int r_max):
    cdef:
        int the_int
    the_int = rand() % (r_max - r_min + 1) + r_min
    return the_int

#cdef list rand_sample(int r_min, int r_max, int number):
#    cdef:
#        list list_int
#    srand (time(NULL))
#    list_int = []
#    for i in range(number):
#        list_int.append(rand() % (r_max - r_min) + r_min)
#    return list_int

cdef class EECO:
    params = {'H': 5,
              'num_stg': 4,
              'FES_MAX' : 20000
            }
    cdef public:
        double[::1] lb, ub
        double[:, ::1] o
        double[:, :] M, M1, M2
        tuple mat
        

    def __init__(self, benchID, D):
        self.lb = np.zeros(D, dtype = np.float64)
        self.ub = np.zeros(D, dtype = np.float64)
        CFunction().get_LB(self.lb, benchID, D)
        CFunction().get_UB(self.ub, benchID, D)
        mat = CFunction().load_mat(benchID, D)
        self.o, self.M, self.M1, self.M2 = (mat[_] for _ in range(4))
        
    ##Initialize population P
    cdef void init_P(self, double[:, ::1] P_X, int NP, int benchID, int D):
        for i in range(NP):
            CFunction().evaluate(benchID, D, P_X[i, :], self.o, self.M, self.M1, self.M2)
    
    ##Initialize population Q      
    cdef list init_Q(self, double[:, ::1] P_X, double[:, ::1] Q_X, int NP, int Lambda, int benchID, int D):
        cdef:
            list sel_idx
        sel_idx = random.sample(range(NP), Lambda)
        for i in range(len(sel_idx)):
            Q_X[i, :] = P_X[sel_idx[i], :]
        return sel_idx
    
    ##Initiallize memory for f and cr
    cdef tuple initMemory(self):
        cdef:
            int H, n, j, num_stg
            list M_CR, M_F, Temp_1, Temp_2
        H = self.params['H']
        num_stg = self.params['num_stg']
        M_CR = []
        M_F = []
        for i in range(num_stg):
            
            Temp_1, Temp_2 = [], []
            for j in range(H):
                Temp_1.append(0.5)  
            M_CR.append(Temp_1)

            for j in range(H):
                Temp_2.append(0.5)   
            M_F.append(Temp_2)
        return M_CR, M_F
    
    ##strategy decision
    cdef int chooseStrategy(self, list ql, list Num_Success_n):
        cdef:
            int n_sum, k, l, Strategy
            double wheel 
        n_sum = 0
        Strategy = 0
        for k in range(4):
            n_sum += Num_Success_n[k] + 2
        if n_sum != 0:
            for k in range(4):
                ql[k] = <double>(Num_Success_n[k] + 2) / n_sum
        for k in range(4):
            if ql[k] < 0.05:
                for l in range(4):
                    ql[l] = 0.25
                    Num_Success_n[l] = 0
                break
        wheel = random.random()
        if wheel <= ql[0]:
            Strategy = 0
        elif wheel <= sum(ql[:2]) and wheel > ql[0]:
            Strategy = 1
        elif wheel <= sum(ql[:3]) and wheel > sum(ql[:2]):
            Strategy = 2
        elif wheel <= sum(ql[:4]) and wheel > sum(ql[:3]):
            Strategy = 3
        return Strategy  
    
    ##generate F and CR in DE
    cdef tuple generate_F_CR(self, int Stg, list Memory_CR, list Memory_F, list Success_CR, list CR, list F):
        cdef:
            int H, i, j, ri, num_stg
            double cr, f
            list muCR, muF
        H = self.params['H']
        num_stg = self.params['num_stg']
        ri = random.randint(0, H - 1)
        muCR = []
        for i in range(num_stg):
            muCR.append(0.5)
        muF = []
        for i in range(num_stg):
            muF.append(0.5)
        if Success_CR[Stg] != []:
            if muCR[Stg] == -1.0 or max(Success_CR[Stg]) == 0.0:
                muCR[Stg] == 0.0
            else:
                muCR[Stg] = Memory_CR[Stg][ri]
        else:
            muCR[Stg] = Memory_CR[Stg][ri]
        muF[Stg] = Memory_F[Stg][ri]
        cr = rand_normal(muCR[Stg], 0.1)
        f = rand_cauchy(muF[Stg], 0.1)
        
        if cr < 0.0:
            cr = 0.0
        elif cr > 1.0:
            cr = 1.0
        while f <= 0.0:
            f = rand_cauchy(muF[Stg], 0.1)
        if f > 1.0:
            f = 1.0
            
        CR.append(cr)
        F.append(f) 
        return cr, f

    
    ##current-to-Qbest/1
    cdef void mutation_1(self, double[:, ::1] QParent, double[:, ::1] QChild, list A, int Lambda, int D, double f, int idx, int bestEqIndex, double grate):
        cdef:
            int x_r1, x_r2
        x_r1 = random.randint(0, Lambda - 1)
        x_r2 = random.randint(0, Lambda + <int>len(A) - 1)
        while x_r1 == idx:
            x_r1 = random.randint(0, Lambda - 1)
        while x_r2 ==x_r1 or x_r2 == idx:
            x_r2 = random.randint(0, Lambda + <int>len(A) - 1)
        for j in range(D):
            if x_r2 < Lambda:
                QChild[idx, j] = QParent[idx, j] + f * (QParent[bestEqIndex, j] - QParent[idx, j]) + f * (QParent[x_r1, j] - QParent[x_r2, j])
            else:
                QChild[idx, j] = QParent[idx, j] + f * (QParent[bestEqIndex, j] - QParent[idx, j]) + f * (QParent[x_r1, j] - A[x_r2 - Lambda][j])
            if QChild[idx, j] < self.lb[j]:
                QChild[idx, j] = min(self.ub[j], 2 * self.lb[j] - QChild[idx, j]) 
            elif QChild[idx, j] > self.ub[j]:
                QChild[idx, j] = max(self.lb[j], 2 * self.ub[j] - QChild[idx, j])
        
    ##randr1/1
    cdef void mutation_2(self, double[:, ::1] QParent, double[:, ::1] QChild, int Lambda, int D, double f, int idx):
        cdef:
            int x_1, x_2, x_3

        x_1 = random.randint(0, Lambda - 1)
        x_2 = random.randint(0, Lambda - 1)
        x_3 = random.randint(0, Lambda - 1)
        while x_1 == x_2:
            x_2 = random.randint(0, Lambda - 1)
        while x_1 == x_3 or x_2 == x_3:
            x_3 = random.randint(0, Lambda - 1)
        for j in range(D):
            QChild[idx, j] = QParent[x_1, j] + f * (QParent[x_2, j] - QParent[x_3, j]) 
            if QChild[idx, j] < self.lb[j]:
                QChild[idx, j] = min(self.ub[j], 2 * self.lb[j] - QChild[idx, j]) 
            elif QChild[idx, j] > self.ub[j]:
                QChild[idx, j] = max(self.lb[j], 2 * self.ub[j] - QChild[idx, j])

    
    
    ##binomial crossover
    cdef void crossover_1(self, double[:, ::1] QParent, double[:, ::1] QChild, int D, double cr, int idx):
        cdef:
            int jRand, j
        jRand = random.randint (0, D - 1)
        for j in range(D):
            if jRand != j and random.random() <= cr:
                QChild[idx, j] = QParent[idx, j]

    ##exponential crossover
    cdef void crossover_2(self, double[:, ::1] QParent, double[:, ::1] QChild, int D, double cr, int idx):
        cdef:
            int n, L
        n = random.randint (0, D - 1)
        L = 0
        while random.random() <= cr and L < D:
            QChild[idx, (n + L) % D] = QParent[idx, (n + L) % D]
            L += 1 
    
    ##DE
    cdef tuple DE(self, double[:, ::1] QParent, double[:, ::1] QChild, list A, int Lambda, int D, int Stg, int idx, int bestEqIndex, double f, double cr, double grate):
        if Stg == 0:
            self.mutation_1(QParent, QChild, A, Lambda, D, f, idx, bestEqIndex, grate)
            self.crossover_1(QParent, QChild, D, cr, idx)
        elif Stg == 1:
            self.mutation_1(QParent, QChild, A, Lambda, D, f, idx, bestEqIndex, grate)
            self.crossover_2(QParent, QChild, D, cr, idx)
        elif Stg == 2:
            self.mutation_2(QParent, QChild, Lambda, D, f, idx)
            self.crossover_1(QParent, QChild, D, cr, idx)
        elif Stg == 3:
            self.mutation_2(QParent, QChild, Lambda, D, f, idx)
            self.crossover_2(QParent, QChild, D, cr, idx)

    ##equivalent function - feasible rule
    cdef void Equ_FR(self, double[:, ::1] Q, int Lambda, int D):
        cdef:
            int jugg, len_Q
            double Fea_min, IF_min
        jugg = 1
        Fea_max = -RAND_MAX
        len_Q = Q.shape[0]
        for i in range(len_Q):
            if Q[i, D + 1] > 0.0:
                jugg = -1
                break
        if jugg == -1:
            for i in range(len_Q):
                if Q[i, D + 1] == 0.0:
                    jugg = 0
                    break
        if jugg == 1:
            for i in range(len_Q):
                Q[i, D + 2] = Q[i, D]
        elif jugg == -1:
            for i in range(len_Q):
                Q[i, D + 2] = Q[i, D + 1]
        else:
            for i in range(len_Q):
                if Q[i, D + 1] == 0.0:
                    if Fea_max > Q[i, D]:
                        Fea_max = Q[i, D]
            for i in range(len_Q):
                if Q[i, D + 1] > 0.0:
                    Q[i, D + 2] = Fea_max + Q[i, D + 1]
                else:
                    Q[i, D + 2] = Q[i, D]

    ##new equivalent function
    cdef void Equ_New(self, double[:, ::1] Q, int Lambda, int D):
        len_Q = Q.shape[0]
        bestIndex = self.find_FR_Best(Q, len_Q, D)
        for i in range(len_Q):
            Q[i, D + 2] = abs(Q[i, D] - Q[bestIndex, D])

    ##calculate fi(x) = w1 * e(x) + w2 * v(x) + w3 * f(x)
    cdef void Hl_Eq(self, double[:, ::1] Q, int Lambda, int D, int idx, double para, double gamma):
        cdef:
            double f_max, f_min, v_max, v_min, eq_max, eq_min, w_t, n, w_i, w1, w2, w3
            int len_Q
        ##normalization
        f_max = self.np_max(Q, D)
        f_min = self.np_min(Q, D)
        v_max = self.np_max(Q, D + 1)
        v_min = self.np_min(Q, D + 1)
#        self.EquFR(Q, Lambda, D)
        self.Equ_New(Q, Lambda, D)
        eq_max = self.np_max(Q, D + 2)
        eq_min = self.np_min(Q, D + 2)
        w_t = para  
        w_i = (<double>idx + 1)/ Lambda

        w1 = w_t * w_i
        w2 = w_t * w_i + gamma
        w3 = (1.0 - w_t) * (1.0 - w_i) 
        len_Q = Q.shape[0]
        for _ in range(len_Q):
            Q[_, D + 3] = w1 * (Q[_, D + 2] - eq_min) / (eq_max - eq_min + 10.0**-100)
            Q[_, D + 3] += w2 * (Q[_, D + 1] - v_min) / (v_max - v_min + 10.0**-100)
            Q[_, D + 3] += w3 * (Q[_, D] - f_min) / (f_max - f_min + 10.0**-100)

    ##minimum value of e, v or f with axis_n  
    cdef double np_min(self, double[:, ::1] P, int axis_n):
        cdef:
            double min_value
            Py_ssize_t bound
        min_value = P[0, axis_n]
        bound = P.shape[0]
        for i in range(bound):
            if min_value > P[i, axis_n]:
                min_value = P[i, axis_n]
        return min_value
    
    ##maximum value of e, v or f with axis_n
    cdef double np_max(self, double[:, ::1] P, int axis_n):
        cdef:
            double max_value
            Py_ssize_t bound
        max_value = P[0, axis_n]
        bound = P.shape[0]
        for i in range(bound):
            if max_value < P[i, axis_n]:
                max_value = P[i, axis_n]
        return max_value
    
    ##selection/updating subpopulation Q
    cdef void selection(self, double[:, ::1] QSum, double[:, ::1] QParent, double[:, ::1] QChild, list A, int Lambda, int D, int idx, int stg, list CR, list F, list Success_cr, list Success_F, list fit_improve, list Num_Success_n):
        if QSum[idx, D + 3] > QSum[Lambda, D + 3]:
            Success_cr[stg].append(CR[len(CR) - 1])
            Success_F[stg].append(F[len(F) - 1])
            Num_Success_n[stg] += 1

            fit_improve[stg].append(QSum[idx, D + 3] - QSum[Lambda, D + 3])
            A.append(QParent[idx, :])
            QParent[idx, :] = QChild[idx, :]
    
    ##|A| < |A|_max
    cdef void stableA(self, list A, int ASize):
        if len(A) > ASize:
            for i in range(<int>len(A) - ASize):
                A.remove(A[random.randint(0, <int>len(A) - 1)])                

    ##Updateing Memory
    cdef void UpdateMemory(self, list Memory_cr, list Memory_F, list Success_cr, list Success_F, list fit_improve, int H, list pos):
        cdef:
            int n, k, num_Scr, num_SF, num_stg
            double f1, f3, f4, weight_1, weight_2, meanScr, meanSF
            
        num_stg = self.params['num_stg']
        for k in range(num_stg):
            if Success_cr[k] != [] and Success_F[k] != []:
                num_Scr = <int>len(Success_cr[k])
                num_SF = <int>len(Success_F[k])
                meanScr = 0.0
                meanSF = 0.0
                weight_1 = 0.0
                f1 = 0.0
                for i in range(num_Scr):
                    weight_1 += abs(fit_improve[k][i])
                for i in range(num_Scr):
                    f1 += abs(fit_improve[k][i]) / (weight_1 + 10.0**-100) * (Success_cr[k][i])
                meanScr = f1 
#                Memory_cr[k][pos[k]] = (meanScr + Success_cr[k][num_Scr - 1]) / 2
                Memory_cr[k][pos[k]] = meanScr
                weight_2 = 0.0
                f3 = 0.0
                f4 = 0.0
                for i in range(num_SF):
                    weight_2 += abs(fit_improve[k][i])
                for i in range(num_SF):
                    f3 += abs(fit_improve[k][i]) / (weight_2 + 10.0**-100) * np.power(Success_F[k][i], 2)
                    f4 += abs(fit_improve[k][i]) / (weight_2 + 10.0**-100) * Success_F[k][i]
                meanSF = f3 / (f4 + 10.0**-100)
#                Memory_F[k][pos[k]] = (meanSF + Success_F[k][num_SF - 1]) / 2
                Memory_F[k][pos[k]] = meanSF
                
                pos[k] = pos[k] + 1
                if pos[k] > H - 1:
                    pos[k] = 0
    
    ##the index of best solution of feasibility rule
    cdef int find_FR_Best(self, double[:, ::1] P, int NP, int D):
        cdef:
            int bestIndex
        bestIndex = 0
        for i in range(NP):
            if P[bestIndex, D + 1] > P[i, D + 1]:
                bestIndex = i
            elif P[bestIndex, D + 1] == P[i, D + 1] and P[bestIndex, D] > P[i, D]:
                bestIndex = i
        return bestIndex
    

    ##the index of best solution of e, v or f or fi
    cdef int findBest(self, double[:, ::1] Q, int Lambda, int D, int axis_n):
        cdef:
            int bestIndex
        bestIndex = 0
        for i in range(Lambda):
            if Q[bestIndex, D + axis_n] > Q[i, D + axis_n]:
                bestIndex = i
        return bestIndex

    ##the main process of HECO-DE
    cdef tuple _optimize(self, int benchID, int D, int Lambda, double Gamma):
        
        cdef:
            int NP, NPinit, NPmin, NPlast, H, num_stg, FES, FES_MAX, gen_count, stg, bestEqIndex, bestIndex, bestFIndex, worstIndex, worstFIndex
            list sel_idx, A, M_CR, M_F, CR, F, Num_Success_n, pos, ql, S_CR, S_F, fit_improve
            double para, cr, f
            double[:, ::1] P, QParent, QChild
            tuple Init_M

        NPinit = 12 * D
        NP = NPinit
#        Lambda = 40
        NPmin = Lambda
        NPlast = 0
        H = self.params['H']
        num_stg = self.params['num_stg']
        FES = NP
        gen_count = 1
        FES_MAX = 20000 * D
        ##population P, 0~D-1 are the solutions, D~D+7 are solution's value of f, v, e, fi, c1, c2 and c3 respectively
        narr_P = np.zeros((NP, D + 7), dtype = np.float64) 
        P = narr_P ##initialize solutions
        for i in range(NP):
            for j in range(D):
                P[i, j] = self.lb[j] + random.random() * (self.ub[j] - self.lb[j])
        self.init_P(P, NP, benchID, D)
        narr_QParent = np.zeros((Lambda, D + 7), dtype = np.float64)
        narr_QChild = np.zeros((Lambda, D + 7), dtype = np.float64)
        narr_QSum = np.zeros((Lambda + 1, D + 7), dtype = np.float64)
        QParent = narr_QParent
        QChild = narr_QChild
        QSum = narr_QSum ##QSum is Q + i_th child
        bestEqIndex = 0

        A = [] ## archive A
        Init_M = self.initMemory()
        M_CR = Init_M[0]
        M_F = Init_M[1]
        CR = []
        F = []
        Num_Success_n = []
        pos = []
        ql = []
        for i in range(num_stg):
            Num_Success_n.append(0)
            pos.append(0)
            ql.append(0.25)
        while FES < FES_MAX:
            ASize = round(4.0 * NP)
            S_CR = []
            S_F = []
            fit_improve = []
            for i in range(num_stg):
                S_CR.append([])
                S_F.append([])
                fit_improve.append([])

            sel_idx = self.init_Q(P, QParent, NP, Lambda, benchID, D)
            para = <double>FES / FES_MAX ##t/T_MAX
            for idx in range(Lambda):
                stg = self.chooseStrategy(ql, Num_Success_n)
                f_cr = self.generate_F_CR(stg, M_CR, M_F, S_CR, CR, F)
                cr = f_cr[0]
                f = f_cr[1]
                self.Hl_Eq(QParent, Lambda, D, idx, para, Gamma)
                bestEqIndex = self.findBest(QParent, Lambda, D, 3)
                self.DE(QParent, QChild, A, Lambda, D, stg, idx, bestEqIndex, f, cr, para)
                CFunction().evaluate(benchID, D, QChild[idx, :], self.o, self.M, self.M1, self.M2)
                QSum[:Lambda, :] = QParent[:, :]
                QSum[Lambda, :] = QChild[idx, :]
                self.Hl_Eq(QSum, Lambda, D, idx, para, Gamma)
                self.selection(QSum, QParent, QChild, A, Lambda, D, idx, stg, CR, F, S_CR, S_F, fit_improve, Num_Success_n)
                self.stableA(A, ASize)
                FES += 1
            for i in range(len(sel_idx)):
                P[sel_idx[i], :] = QParent[i, :]
            self.UpdateMemory(M_CR, M_F, S_CR, S_F, fit_improve, H, pos)
            gen_count += 1
            
            ##get the FES needed for arriving sucess condition
            bestIndex = self.find_FR_Best(P, NP, D)
            if P[bestIndex, D] < 0.0001 and P[bestIndex, D + 1] == 0.0 and Success == 0:
                Srun = FES
                Success = 1
            
            ##population size reduction
            NPlast = narr_P.shape[0]
            if NP > Lambda:
                NP = round(<double>(NPmin - NPinit) / (FES_MAX) * FES + NPinit)
            if NP < NPlast and NP >= Lambda:
                for i in range(NPlast - NP):
                    r = random.randint(0, narr_P.shape[0] - 1)
                    while r == bestIndex:
                        r = random.randint(0, narr_P.shape[0] - 1)
                    narr_P = np.delete(narr_P, r, 0)
                    P = narr_P
                
        S_CR.clear()
        S_F.clear()
        fit_improve.clear()
        return P[bestIndex, D], P[bestIndex, D + 1], <int>P[bestIndex, D + 4], <int>P[bestIndex, D + 5], <int>P[bestIndex, D + 6]
        
        
    def optimize(self, benchID, D, Lambda, Gamma):
        return self._optimize(benchID, D, Lambda, Gamma)
        
        
        
        
        
        
        
        
        
        
        