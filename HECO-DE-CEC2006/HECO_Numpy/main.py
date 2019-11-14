# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:41:58 2018

@author: burningxt
"""

#from HEOEA_cy import HEOEA
#import GECCO_cy
from HECO_cy import EECO
#from cec2017_cy import CFunction, Individual
from cec2006 import CFunction
import timeit
import multiprocessing as mp
import os
import numpy as np
import xlwt 
from xlwt import Workbook 
        


#########################CEC2006#####################################
def run(runs):
    for Lambda in [30, 35, 40, 45, 50]:
        for Gamma in [0.7]:
#    for Lambda in [45]:
#        for Gamma in [0.5, 0.6, 0.8, 0.9]:
            for benchID in range(1, 25):
                D = CFunction().getParam(benchID)
                results = EECO(benchID, D).optimize(benchID, D, Lambda, Gamma)
                print(results[0], file = open("outputs/lambda = {}/F{}_obj.txt".format(Lambda, benchID), "a"))
                print(results[1], file = open("outputs/lambda = {}/F{}_vio.txt".format(Lambda, benchID), "a"))
                print(results[2], results[3], results[4], file = open("outputs/lambda = {}/F{}_c.txt".format(Lambda, benchID), "a"))
                print(results[5], file = open("outputs/lambda = {}/Success{}.txt".format(Lambda, benchID), "a"))
                
#                print(results[0], file = open("outputs/Gamma = {}/F{}_obj.txt".format(Gamma, benchID), "a"))
#                print(results[1], file = open("outputs/Gamma = {}/F{}_vio.txt".format(Gamma, benchID), "a"))
#                print(results[2], results[3], results[4], file = open("outputs/Gamma = {}/F{}_c.txt".format(Gamma, benchID), "a"))
#                print(results[5], file = open("outputs/Gamma = {}/Success{}.txt".format(Gamma, benchID), "a"))
   

#def run(runs):
#    for Lambda in [45]:
#        for Gamma in [0.7]:  
#            for eq in ['f']:
#            for eq in ['feasible rule']:
#                for benchID in range(1, 25):
#                    D = CFunction().getParam(benchID)
#                    results = EECO(benchID, D).optimize(benchID, D, Lambda, Gamma)
#                    print(results[0], file = open("outputs/eq = {}/F{}_obj.txt".format(eq, benchID), "a"))
#                    print(results[1], file = open("outputs/eq = {}/F{}_vio.txt".format(eq, benchID), "a"))
#                    print(results[2], results[3], results[4], file = open("outputs/lambda = {}/F{}_c.txt".format(eq, benchID), "a"))
#                    print(results[5], file = open("outputs/eq = {}/Success{}.txt".format(eq, benchID), "a"))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    start = timeit.default_timer()
    pool = mp.Pool(processes = 5)
    res = pool.map(run, range(25))
    stop = timeit.default_timer()
    print('Time: ', stop - start)


