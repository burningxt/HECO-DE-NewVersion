# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 01:33:17 2019

@author: burningxt
"""

import os
import numpy as np
import xlwt 
import numpy as np
from xlwt import Workbook 

def getData(folder, benchID, obj_arr, vio_arr, succ_arr):
    cur_path = os.path.dirname(__file__)
    os.listdir()
    new_path_1 = os.path.relpath('HECO_Numpy/outputs/{}/F{}_obj.txt'.format(folder, benchID), cur_path)   
    with open(new_path_1, 'r') as f:
        _ = 0
        for line in f:
            obj_arr[benchID - 1, _] = float(line)
            _ += 1
    new_path_1 = os.path.relpath('HECO_Numpy/outputs/{}/Success{}.txt'.format(folder, benchID), cur_path)   
    with open(new_path_1, 'r') as f:
        _ = 0
        for line in f:
            succ_arr[benchID - 1, _] = float(line)
            _ += 1
    new_path_1 = os.path.relpath('HECO_Numpy/outputs/{}/F{}_vio.txt'.format(folder, benchID), cur_path)   
    with open(new_path_1, 'r') as f:
        _ = 0
        for line in f:
            vio_arr[benchID - 1, _] = float(line)
            _ +=1

#for folder in ['lambda = 35', 'lambda = 40', 'lambda = 45', 'lambda = 50', 'lambda = 55']:
for folder in ['Gamma = 0.5', 'Gamma = 0.6', 'Gamma = 0.7', 'Gamma = 0.8', 'Gamma = 0.9']:
#for folder in ['eq = f', 'eq = feasible rule', 'eq = neweq']:
    obj_arr = np.zeros((24, 25))
    vio_arr = np.zeros((24, 25))
    succ_arr = np.zeros((24, 25))
    for benchID in range(1, 25):
        getData(folder, benchID, obj_arr, vio_arr, succ_arr)
#    print(vio_arr)
#print(obj_arr)

    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1') 
    sheet1.write(0, 1, 'Mean OFV')
    sheet1.write(0, 2, 'SR')
    sheet1.write(0, 3, 'FR')
    for benchID in range(1, 25):
        sheet1.write(benchID, 0, 'F{}'.format(benchID)) 
        sheet1.write(benchID, 1, np.mean(obj_arr[benchID - 1])) 
        sheet1.write(benchID, 2, np.count_nonzero(succ_arr[benchID - 1, :] != 0) * 4)
        sheet1.write(benchID, 3, np.count_nonzero(vio_arr[benchID - 1, :] == 0.0) * 4)
    wb.save('result_analysis/{}/mean_obj.xls'.format(folder))