# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:34:13 2018

@author: burningxt
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

 
setup(
#    ext_modules = cythonize("cec2017_cy.pyx")
#    ext_modules = cythonize("EECO_cy.pyx", annotate=True)
#    ext_modules = cythonize("temp.pyx", annotate=True)
    ext_modules = cythonize((Extension("HECO_cy", sources=["HECO_cy.pyx"], include_dirs=[np.get_include()], ), ), annotate=True)
#    ext_modules = cythonize((Extension("cec2017_cy", sources=["cec2017_cy.pyx"], include_dirs=[np.get_include()], ), ), annotate=True)
)