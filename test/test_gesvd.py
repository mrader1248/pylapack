# -*- coding: utf-8 -*-
"""
Created on 2017-12-14T13:11:18.771Z

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
from time import time

import pylapack
pylapack.init()

print("t    m    n   time      err")

for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
    for m in [1024, 2048]:
        for n in [m // 4, m // 2, m]:

            a = np.asarray(np.random.rand(m, n), dtype)
            if np.iscomplexobj(a):
                a = a + 1j*np.asarray(np.random.rand(m, n), dtype)
            a = np.asfortranarray(a)

            t = time()
            svd = pylapack.GESVD(dtype, m, n)
            u,s,v = svd.run(a, False)
            t = time() - t

            err = np.max(np.abs(a - np.dot(u*s, v)))

            dtstr = {
                np.float32: "s", np.float64: "d",
                np.complex64: "c", np.complex128: "z"
            }[dtype]
            msg = "{:s} {:4d} {:4d} {:6.2f} {:8.2e}"
            msg = msg.format(dtstr, m, n, t, err)
            print(msg)
