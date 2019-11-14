# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:41:58 2018

@author: burningxt
"""

#from EECO_np import EECO
from HECO_cy import EECO
import timeit
import multiprocessing as mp
        
def run(runs):
    for Lambda in [15, 20, 35, 30, 35]:
        for Gamma in [0.1]:
#    for Lambda in [20]:
#        for Gamma in [0.0, 0.2, 0.3, 0.4]:
            for D in [10, 30, 50, 100]:
                for benchID in range(1, 29):
                    results = EECO(benchID, D).optimize(benchID, D, Lambda, Gamma)
                    print(results[0], file = open("Results/lambda = {}/F{}_{}D_obj.txt".format(Lambda, benchID, D), "a"))
                    print(results[1], file = open("Results/lambda = {}/F{}_{}D_vio.txt".format(Lambda, benchID, D), "a"))
                    print(results[2], results[3], results[4], file = open("Results/lambda = {}/F{}_{}D_c.txt".format(Lambda, benchID, D), "a")) 
                   
#                    print(results[0], file = open("outputs/gamma = {}/F{}_{}D_obj.txt".format(Gamma, benchID, D), "a"))
#                    print(results[1], file = open("outputs/gamma = {}/F{}_{}D_vio.txt".format(Gamma, benchID, D), "a"))
#                    print(results[2], results[3], results[4], file = open("outputs/gamma = {}/F{}_{}D_c.txt".format(Gamma, benchID, D), "a")) 


#def run(runs):
#    for Lambda in [20]:
#        for Gamma in [0.1]:
#            for eq in ['f']:
#            for eq in ['feasible rule']:
#                for D in [10, 30, 50, 100]:
#                    for benchID in range(1, 29):
#                        results = EECO(benchID, D).optimize(benchID, D, Lambda, Gamma)
#                        print(results[0], file = open("outputs/eq = {}/F{}_{}D_obj.txt".format(eq, benchID, D), "a"))
#                        print(results[1], file = open("outputs/eq = {}/F{}_{}D_vio.txt".format(eq, benchID, D), "a"))
#                        print(results[2], results[3], results[4], file = open("outputs/eq = {}/F{}_{}D_c.txt".format(eq, benchID, D), "a")) 
 
    
                   
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start = timeit.default_timer()
    pool = mp.Pool(processes = 5)
    res = pool.map(run, range(25))
    stop = timeit.default_timer()
    print('Time: ', stop - start)   

    