# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 23:21:55 2019

@author: burningxt
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import e

Lambda = 30
cases = np.arange(0.0/Lambda, 11.01/Lambda, 1.0/Lambda)

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']
        
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.3))
w_t = np.arange(0.0, 1.01, 0.1)

gamma = 0.7
y11 = w_t * 1/40
y21 = w_t * 1/40 + gamma
y31 = (1 - w_t) * (1 - 1/40)
y12 = w_t * 20/40
y22 = w_t * 20/40 + gamma
y32 = (1 - w_t) * (1 - 20/40)
y13 = w_t * 40/40
y23 = w_t * 40/40 + gamma
y33 = (1 - w_t) * (1 - 40/40)

#gamma = 0.1
#y11 = w_t * 1/20
#y21 = w_t * 1/20 + gamma
#y31 = (1 - w_t) * (1 - 1/20)
#y12 = w_t * 10/20
#y22 = w_t * 10/20 + gamma
#y32 = (1 - w_t) * (1 - 10/20)
#y13 = w_t * 20/20
#y23 = w_t * 20/20 + gamma
#y33 = (1 - w_t) * (1 - 20/20)


axes[0].plot(w_t, y11 / (y11 + y21 + y31), ls='-', marker = '^', label='$i = 1$') 
axes[0].plot(w_t, y12 / (y12 + y22 + y32), ls='-', marker = '*', label='$i = \lambda/2$')
axes[0].plot(w_t, y13 / (y13 + y23 + y33), ls='-', marker = '+', label='$i = \lambda$') 
axes[0].set_xlabel("$t/T_{\max}$")
axes[0].set_ylabel("$w_{1i,t}$")
axes[0].legend(loc='upper left')
axes[0].set_ylim(-0.05, 1.05)

#axes[1].set_title("$w_{2i,t}$")
axes[1].plot(w_t, y21 / (y11 + y21 + y31), ls='-', marker = '^', label='$i = 1$') 
axes[1].plot(w_t, y22 / (y12 + y22 + y32), ls='-', marker = '*', label='$i = \lambda/2$')
axes[1].plot(w_t, y23 / (y13 + y23 + y33), ls='-', marker = '+', label='$i = \lambda$') 
axes[1].set_xlabel("$t/T_{\max}$")
axes[1].set_ylabel("$w_{2i,t}$")
axes[1].legend(loc='lower right')
axes[1].set_ylim(-0.05, 1.05)

#axes[2].set_title("$w_{3i,t}$")
axes[2].plot(w_t, y31 / (y11 + y21 + y31), ls='-', marker = '^', label='$i = 1$') 
axes[2].plot(w_t, y32 / (y12 + y22 + y32), ls='-', marker = '*', label='$i = \lambda/2$')
axes[2].plot(w_t, y33 / (y13 + y23 + y33), ls='-', marker = '+', label='$i = \lambda$') 
axes[2].set_xlabel("$t/T_{\max}$")
axes[2].set_ylabel("$w_{3i,t}$")
axes[2].legend(loc='upper right')
axes[2].set_ylim(-0.05, 1.05)


#axes[0, 0].plot(w_t, y11, ls='-', marker = '^', label='$i = 1$') 
#axes[0, 0].plot(w_t, y13, ls='-', marker = '*', label='$i = 5$')
#axes[0, 0].plot(w_t, y12, ls='-', marker = '+', label='$i = \lambda$') 
#axes[0, 0].set_xlabel("$t/T_{\max}$")
#axes[0, 0].set_ylabel("$w_{1i,t}$")
##axes[0].legend(loc='upper left')
#axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
#
##axes[1].set_title("$w_{2i,t}$")
#axes[0, 1].plot(w_t, y21, ls='-', marker = '^', label='$i = 1$') 
#axes[0, 1].plot(w_t, y23, ls='-', marker = '*', label='$i = 5$')
#axes[0, 1].plot(w_t, y22, ls='-', marker = '+', label='$i = \lambda$') 
#axes[0, 1].set_xlabel("$t/T_{\max}$")
#axes[0, 1].set_ylabel("$w_{2i,t}$")
##axes[0, 1].legend(loc='upper left')
#axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
##axes[2].set_title("$w_{3i,t}$")
#axes[1, 0].plot(w_t, y31, ls='-', marker = '^', label='$i = 1$') 
#axes[1, 0].plot(w_t, y33, ls='-', marker = '*', label='$i = 5$')
#axes[1, 0].plot(w_t, y32, ls='-', marker = '+', label='$i = \lambda$') 
#axes[1, 0].set_xlabel("$t/T_{\max}$")
#axes[1, 0].set_ylabel("$w_{3i,t}$")
#axes[1, 0].legend(loc='upper right')
#axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#
#axes[1, 1].remove()



#plt.title('w1')

plt.ylim(-0.05, 1.05)
fig.tight_layout()
#plt.savefig('fig_wt_2006.pdf', dpi=1000)
plt.show()
