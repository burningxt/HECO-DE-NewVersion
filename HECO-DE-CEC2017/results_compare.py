# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:56:03 2018

@author: burningxt
"""

import os
import numpy as np
import xlwt 
from xlwt import Workbook 


class buildResults:
    cur_path = os.path.dirname(__file__)
    def getData(self, res_arr, folders):
        os.listdir()
        i = 0
        for folder in folders:
            for benchID in range(1, 29):
                j = 0
                for D in [10, 30, 50, 100]:
                    k = 0
                    for attr in ['obj', 'vio', 'c']:
                        new_path = os.path.relpath('HECO_Numpy/outputs/{}/F{}_{}D_{}.txt'.format(folder, benchID, D, attr), self.cur_path)
                        with open(new_path, 'r') as f:
                            l = 0
                            for line in f:
                                if attr != 'c':
                                    res_arr[i, benchID - 1, j, k, l] = float(line)
                                else:
                                    res_arr[i, benchID - 1, j, k, l] = line
                                l += 1
                        k += 1
                    j += 1
            i += 1
    def getExcel(self, mean_fea, res_arr, folders):
        i = 0
        for folder in folders:
            j = 0
            for D in [10, 30, 50, 100]: 
                wb = Workbook() 
                sheet1 = wb.add_sheet('Sheet 1') 
                sheet1.write(0, 0, 'problem') 
                sheet1.write(1, 0, 'Best') 
                sheet1.write(2, 0, 'Median')
                sheet1.write(3, 0, 'c')
                sheet1.write(4, 0, 'v')
                sheet1.write(5, 0, 'mean')
                sheet1.write(6, 0, 'Worst')
                sheet1.write(7, 0, 'std')
                sheet1.write(8, 0, 'SR')
                sheet1.write(9, 0, 'vio')
                
                for benchID in range(1, 29):
                    fea_num = 0
                    for run_idx in range(25):
                        if res_arr[i, benchID - 1, j, 1, run_idx] == 0.0:
                            fea_num += 1
                    
                    res_arr_trans = res_arr[i, benchID - 1, j, :, :].transpose()
                    sorted_res_arr = res_arr_trans[np.lexsort((res_arr_trans[:, 0], res_arr_trans[:, 1]))]
                    sheet1.write(0, benchID, 'C0{}'.format(benchID))
                    sheet1.write(1, benchID, "{0:.{1}e}".format(sorted_res_arr[0, 0], 8))
                    sheet1.write(2, benchID, "{0:.{1}e}".format(sorted_res_arr[12, 0], 8))
                    sheet1.write(3, benchID, "{}".format(sorted_res_arr[12, 2]))
                    sheet1.write(4, benchID, "{0:.{1}e}".format(sorted_res_arr[12, 1], 8))
                    sheet1.write(5, benchID, "{0:.{1}e}".format(np.sum(res_arr[i, benchID - 1, j, 0, :]) / 25, 8))
                    sheet1.write(6, benchID, "{0:.{1}e}".format(sorted_res_arr[24, 0], 8))
                    sheet1.write(7, benchID, "{0:.{1}e}".format(np.std(res_arr[i, benchID - 1, j, 0, :]), 8))
                    sheet1.write(8, benchID, "{}".format(fea_num / 25))
                    sheet1.write(9, benchID, "{0:.{1}e}".format(np.sum(res_arr[i, benchID - 1, j, 1, :]) / 25, 8))     
                wb.save('result_analysis/{}/full_results{}D.xls'.format(folder, D)) 
                j += 1
            i += 1
            
class Mean:
    def fea_obj_vio(self, mean_fea, mean_obj, mean_vio, res_arr, folders):     
        i = 0
        for folder in folders:
            j = 0
            for D in [10, 30, 50, 100]: 
                for benchID in range(1, 29):
                    fea_num = 0
                    for run_idx in range(25):
                        if res_arr[i, benchID - 1, j, 1, run_idx] == 0.0:
                            fea_num += 1
                    mean_fea[i, j, benchID - 1] = fea_num / 25
                    mean_obj[i, j, benchID - 1] = np.sum(res_arr[i, benchID - 1, j, 0, :]) / 25
                    mean_vio[i, j, benchID - 1] = np.sum(res_arr[i, benchID - 1, j, 1, :]) / 25
                file = open("result_analysis/{}/mean_fea_{}D.txt".format(folder, D), "w")
                for benchID in range(1, 29):
                    file.write(str('%.8f'%round(100 * mean_fea[i, j, benchID - 1], 8)) + '\n')
                file.close()
                file = open("result_analysis/{}/mean_obj_{}D.txt".format(folder, D), "w")
                for benchID in range(1, 29):
                    file.write(str('%.8f'%round(mean_obj[i, j, benchID - 1], 8)) + '\n')
                file.close()
                file = open("result_analysis/{}/mean_vio_{}D.txt".format(folder, D), "w")
                for benchID in range(1, 29):
                    file.write(str('%.8f'%round(mean_vio[i, j, benchID - 1], 8)) + '\n')
                file.close()
                j += 1
            i +=1

class Median:
    def obj_vio(self, median_obj, median_vio, res_arr, folders):     
        i = 0
        for folder in folders:
            j = 0
            for D in [10, 30, 50, 100]: 
                for benchID in range(1, 29):
                    res_arr_trans = res_arr[i, benchID - 1, j, :, :].transpose()
#                    print(res_arr[i, benchID - 1, j, :, :])
#                    print(res_arr_trans)
                    sorted_res_arr = res_arr_trans[np.lexsort((res_arr_trans[:, 0], res_arr_trans[:, 1]))]
#                    print(sorted_res_arr)
                    median_obj[i, j, benchID - 1] = sorted_res_arr[12, 0]
                    median_vio[i, j, benchID - 1] = sorted_res_arr[12, 1]
                file = open("result_analysis/{}/median_obj_{}D.txt".format(folder, D), "w")
                for benchID in range(1, 29):
                    file.write(str('%.8f'%round(median_obj[i, j, benchID - 1], 8)) + '\n')
                file.close()
                file = open("result_analysis/{}/median_vio_{}D.txt".format(folder, D), "w")
                for benchID in range(1, 29):
                    file.write(str('%.8f'%round(median_vio[i, j, benchID - 1], 8)) + '\n')
                file.close()
                j += 1
            i +=1

#folders = ['eq/eq = f', 'eq/eq = feasible rule', 'eq/eq = neweq']
#folders = ['lambda/lambda = 15', 'lambda/lambda = 20', 'lambda/lambda = 25', 'lambda/lambda = 30', 'lambda/lambda = 35']
folders = ['gamma/gamma = 0.0', 'gamma/gamma = 0.1', 'gamma/gamma = 0.2', 'gamma/gamma = 0.3', 'gamma/gamma = 0.4']
res_arr = np.empty((len(folders), 28, 4, 3, 25), dtype = object)
mean_fea = np.empty((len(folders), 4, 28))
mean_obj = np.empty((len(folders), 4, 28))
mean_vio = np.empty((len(folders), 4, 28))
median_obj = np.empty((len(folders), 4, 28))
median_vio = np.empty((len(folders), 4, 28))
buildResults().getData(res_arr, folders)
Mean().fea_obj_vio(mean_fea, mean_obj, mean_vio, res_arr, folders)
Median().obj_vio(median_obj, median_vio, res_arr, folders)
buildResults().getExcel(mean_fea, res_arr, folders)

   
    





















