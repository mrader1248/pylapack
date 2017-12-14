# -*- coding: utf-8 -*-
"""
Created on 2017-12-13T14:28:57.331Z

@author: mrader1248
"""

########## python 2/3 compatibility ##########
from __future__ import absolute_import, division, print_function
import sys
if sys.version_info.major == 2:
    range = xrange
    xrange = None # to prevent accidental usage
##############################################

import ctypes
import numpy as np

def init(libpath="liblapack.so"):
    global libblas
    liblapack = ctypes.cdll.LoadLibrary(libpath)

    global _sgesvd
    _sgesvd = liblapack.sgesvd_
    _sgesvd.argtypes = [ctypes.c_void_p]*14
    # JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO

    global _dgesvd
    _dgesvd = liblapack.dgesvd_
    _dgesvd.argtypes = [ctypes.c_void_p]*14
    # JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO

    global _cgesvd
    _cgesvd = liblapack.cgesvd_
    _cgesvd.argtypes = [ctypes.c_void_p]*15
    # JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, INFO

    global _zgesvd
    _zgesvd = liblapack.zgesvd_
    _zgesvd.argtypes = [ctypes.c_void_p]*15
    # JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, RWORK, INFO

    global _gesvd
    _gesvd = {
        np.float32: _sgesvd,
        np.float64: _dgesvd,
        np.complex64: _cgesvd,
        np.complex128: _zgesvd
    }



    global _sgesvdx
    _sgesvdx = liblapack.sgesvdx_
    _sgesvdx.argtypes = [ctypes.c_void_p]*21
    # JOBU, JOBVT, RANGE, M, N, A, LDA, VL, VU, IL, IU, NS, S, U, LDU, VT, LDVT,
    # WORK, LWORK, IWORK, INFO

    global _dgesvdx
    _dgesvdx = liblapack.dgesvdx_
    _dgesvdx.argtypes = [ctypes.c_void_p]*21
    # JOBU, JOBVT, RANGE, M, N, A, LDA, VL, VU, IL, IU, NS, S, U, LDU, VT, LDVT,
    # WORK, LWORK, IWORK, INFO

    global _cgesvdx
    _cgesvdx = liblapack.cgesvdx_
    _cgesvdx.argtypes = [ctypes.c_void_p]*22
    # JOBU, JOBVT, RANGE, M, N, A, LDA, VL, VU, IL, IU, NS, S, U, LDU, VT, LDVT,
    # WORK, LWORK, RWORK, IWORK, INFO

    global _zgesvdx
    _zgesvdx = liblapack.zgesvdx_
    _zgesvdx.argtypes = [ctypes.c_void_p]*22
    # JOBU, JOBVT, RANGE, M, N, A, LDA, VL, VU, IL, IU, NS, S, U, LDU, VT, LDVT,
    # WORK, LWORK, RWORK, IWORK, INFO

    global _gesvdx
    _gesvdx = {
        np.float32: _sgesvdx,
        np.float64: _dgesvdx,
        np.complex64: _cgesvdx,
        np.complex128: _zgesvdx
    }
