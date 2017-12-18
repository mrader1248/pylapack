# -*- coding: utf-8 -*-
"""
Created on 2017-12-14T10:37:35.969Z

@author: mrader1248
"""

########## python 2/3 compatibility ##########
from __future__ import absolute_import, division, print_function
import sys
if sys.version_info.major == 2:
    range = xrange
    xrange = None # to prevent accidental usage
##############################################

import os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import scipy.linalg as la
from time import time

import pylapack
pylapack.init()

print("t    m    n    k   time      err")

for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
    for m in [1024, 2048]:
        for n in [m // 4, m // 2, m]:
            for k in [n // 16, n // 4, n]:

                a = np.asarray(np.random.rand(m, n), dtype)
                if np.iscomplexobj(a):
                    a = a + 1j*np.asarray(np.random.rand(m, n), dtype)
                a = np.asfortranarray(a)

                s2 = la.svd(a, full_matrices=False)[1][:k]

                t = time()
                svd = pylapack.GESVDX(dtype, m, n, svals=(0,k-1))
                u,s,v = svd.run(a, False)
                t = time() - t

                err = np.max(np.abs(s - s2))

                dtstr = {
                    np.float32: "s", np.float64: "d",
                    np.complex64: "c", np.complex128: "z"
                }[dtype]
                msg = "{:s} {:4d} {:4d} {:4d} {:6.2f} {:8.2e}"
                msg = msg.format(dtstr, m, n, k, t, err)
                print(msg)
