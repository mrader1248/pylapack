# -*- coding: utf-8 -*-
"""
Created on 2017-12-18T10:33:39.414Z

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

#n = np.asarray(np.logspace(10, 14, 5+4*2, base=2), int)
n = np.asarray(np.logspace(10, 12, 3+2, base=2), int)
n -= (n % 16)
k = [128,256,512,1024]
dtype = np.complex128

t = {"gesvd": [], "gesdd": [], "gesvdx": [[] for _ in range(len(k))]}

for j in range(len(n)):
    a = np.asarray(np.random.rand(n[j], n[j]), dtype)
    if np.iscomplexobj(a):
        a = a + 1j*np.asarray(np.random.rand(n[j], n[j]), dtype)
    a = np.asfortranarray(a)

    t0 = time()
    svd = pylapack.GESVD(dtype, n[j], n[j])
    u,s,v = svd.run(a, False)
    t["gesvd"].append(time() - t0)

    import scipy.linalg as la
    t0 = time()
    u,s,v = la.svd(a, lapack_driver="gesdd")
    t["gesdd"].append(time() - t0)

    for l in range(len(k)):
        t0 = time()
        svd = pylapack.GESVDX(dtype, n[j], n[j], svals=(0,k[l]-1))
        u,s,v = svd.run(a, False)
        t["gesvdx"][l].append(time() - t0)


sys.stdout.write("n     gesvd    gesdd   ")
for l in range(len(k)):
    sys.stdout.write("gesvdx{:d}".format(k[l]))
sys.stdout.write("\n")
for j in range(len(n)):
    sys.stdout.write("{:5d} {:8.3f} {:8.3f}".format(
        t["gesvd"][j], t["gesdd"][j]))
    for l in range(len(k)):
        sys.stdout.write(" {:8.3f}".format(t["gesvdx"][l]))
    sys.stdout.write("\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.plot(n, t["gesvd"], "k-", label="gesvd")
plt.plot(n, t["gesdd"], "k--", label="gesdd")
for l in range(len(k)):
    plt.plot(
        n, t["gesvdx"][l],
        ["b-", "b--", "r-", "r--"][l],
        label="gesvdx, $k={:d}$".format(k[l])
    )
plt.xlabel("$n$")
plt.ylabel("$t$")
plt.xscale("log", basex=2)
plt.yscale("log")
plt.legend(loc="best")
plt.grid()
plt.savefig("benchmark_svd.pdf")
plt.close()
#plt.show()
