# -*- coding: utf-8 -*-
"""
Created on 2017-12-13T11:35:31.520Z

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

class GESVDX:

    """
    With this class the LAPACK functions xGESVDX can be used. They are
    especially useful if one wants to perform a truncated SVD.

    Example
    -------

    # compute truncated rank-k SVD of an m x n matrix:
    a = np.random.rand(m, n).reshape(m, n, order="F")
    svd = GESVDX(a.dtype, m, n, svals=(1, k))
    u,s,v = svd.run(a)
    """

    def __init__(self, dtype, m, n,
                 compute_u=True, compute_v=True,
                 sval_interval=None, svals=None):
        """
        Parameters
        ----------
        dtype : type
            type of matrix to decompose
        m : int
            number of rows of matrix to decompose
        n : int
            number of columns of matrix to decompose
        compute_u : bool, optional
            whether to compute matrix u, default is True
        compute_v : bool, optional
            whether to compute matrix v, default is True
        svals_interval: (lo, hi), optional
            only compute singular values which are in the half-open
            interval (lo,hi]; only svals_interval or svals can be used;
            if neither svals_interval nor svals is defined all singular
            values and vectors are computed
        svals: (lo, hi), optional
            only compute singular values within the index range
            0 <= lo <= hi <= min(m,n) where smaller indices correspond to
            larger singular values; only svals_interval or svals can be used;
            if neither svals_interval nor svals is defined all singular
            values and vectors are computed
        """

        if sval_interval is not None and svals is not None:
            raise ValueError("either sval_interval or svals has to be None")

        self.dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
        self.dtypereal = self.dtype(0).real.dtype.type
        self.complex = np.iscomplexobj(self.dtype(0))

        from .lapack import _gesvdx
        self.gesvdx = _gesvdx[self.dtype]

        self.m = m
        self.n = n
        mn = min(m, n)
        self.maxns = mn

        self.compute_u = compute_u
        self.jobu = ctypes.byref(ctypes.c_char("v" if compute_u else "n"))
        self.compute_v = compute_v
        self.jobv = ctypes.byref(ctypes.c_char("v" if compute_v else "n"))

        self.range = "a"

        if sval_interval is None:
            self.vl = ctypes.c_void_p(0)
            self.vu = ctypes.c_void_p(0)
        else:
            self.range = "v"
            self.vl = np.array(sval_interval[0], dtr)
            self.vl = self.vl.ctypes.data_as(ctypes.c_void_p)
            self.vu = np.array(sval_interval[1], dtr)
            self.vu = self.vu.ctypes.data_as(ctypes.c_void_p)

        if svals is None:
            self.il = ctypes.c_void_p(0)
            self.iu = ctypes.c_void_p(0)
        else:
            self.range = "i"
            self.il = ctypes.byref(ctypes.c_int(svals[0]+1))
            self.iu = ctypes.byref(ctypes.c_int(svals[1]+1))
            self.maxns = svals[1] - svals[0] + 1

        self.range = ctypes.byref(ctypes.c_char(self.range))

        self.lda = self.m

        self.s = np.empty(mn, self.dtypereal)

        if self.compute_u:
            self.u = np.empty((m, self.maxns), self.dtype, "F")
            self.ldu = self.u.strides[1] // self.u.itemsize
            self.u_ptr = self.u.ctypes.data_as(ctypes.c_void_p)
        else:
            self.ldu = 0 # never referenced
            self.u_ptr = ctypes.c_void_p(0)

        if self.compute_v:
            self.v = np.empty((self.maxns, n), self.dtype, "F")
            self.ldv = self.v.strides[1] // self.v.itemsize
            self.v_ptr = self.v.ctypes.data_as(ctypes.c_void_p)
        else:
            self.ldv = 0 # never referenced
            self.v_ptr = ctypes.c_void_p(0)

        self.ns = ctypes.c_int()
        self.info = ctypes.c_int()

        self.iwork = np.empty(12*mn, np.int)

        work = np.empty(1, self.dtype)
        if self.complex:
            self.rwork = np.empty(17*mn**2, self.dtypereal)

            self.gesvdx(
                self.jobu, self.jobv, self.range,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.vl, self.vu, self.il, self.iu,
                ctypes.byref(self.ns),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(-1)),
                self.rwork.ctypes.data_as(ctypes.c_void_p),
                self.iwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )
        else:
            self.gesvdx(
                self.jobu, self.jobv, self.range,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.vl, self.vu, self.il, self.iu,
                ctypes.byref(self.ns),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(-1)),
                self.iwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )
        assert self.info.value == 0

        self.lwork = int(work[0].real)
        self.work = np.empty(self.lwork, self.dtype)

    def run(self, a, overwrite_a=True):
        """
        Performs the defined SVD of the given matrix a.

        Parameters
        ----------
        a : np.ndarray[shape=(m,n)]
            the matrix to decompose; has to be F-ordered
        overwrite_a : bool, optional
            defines whether the matrix a can be overwritten; if False a working
            copy of the matrix a is created; default is True

        Returns
        ----
        u : np.ndarray[shape=(m,min(m,n))]
            isometric matrix containing left singular vectors as columns;
            only returned if compute_u is True
        s : np.ndarray[shape=(min(m,n),)]
            singular values
        v : np.ndarray[shape=(min(m,n),n)]
            hermitian conjugate of isometric matrix containing right singular
            vectors as rows; only returned if compute_v is True
        """

        if not overwrite_a:
            a = a.copy("F")

        assert a.dtype == self.dtype
        assert a.strides[0] == a.itemsize
        assert a.strides[1] == a.itemsize * self.lda

        if self.complex:
            self.gesvdx(
                self.jobu, self.jobv, self.range,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                a.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.vl, self.vu, self.il, self.iu,
                ctypes.byref(self.ns),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                self.work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lwork)),
                self.rwork.ctypes.data_as(ctypes.c_void_p),
                self.iwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )
        else:
            self.gesvdx(
                self.jobu, self.jobv, self.range,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                a.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.vl, self.vu, self.il, self.iu,
                ctypes.byref(self.ns),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                self.work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lwork)),
                self.iwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )

        k = self.ns.value
        if self.compute_u:
            if self.compute_v:
                return self.u[:,:k], self.s[:k], self.v[:k]
            else:
                return self.u[:,:k], self.s[:k]
        else:
            if self.compute_v:
                return self.s[:k], self.v[:k]
            else:
                return self.s[:k]
