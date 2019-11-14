# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:34:13 2018

@author: burningxt
"""

from distutils.core import setup
from Cython.Build import cythonize
 
setup(
   # ext_modules = cythonize("cec2017_cy.pyx")
#   ext_modules = cythonize("cec2006_cy.pyx")
    ext_modules = cythonize("HECO_cy.pyx")
)