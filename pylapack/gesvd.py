# -*- coding: utf-8 -*-
"""
Created on 2017-12-14T11:06:06.503Z

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

class GESVD:

    """
    With this class the LAPACK functions xGESVD can be used. These functions
    are usually slower than xGESDD, but sometimes more stable.

    Example
    -------

    # compute truncated rank-k SVD of an m x n matrix:
    a = np.random.rand(m, n).reshape(m, n, order="F")
    svd = GESVD(a.dtype, m, n)
    u,s,v = svd.run(a, overwrite_a=False)
    assert np.allclose(a, np.dot(u*s, v))
    """

    def __init__(self, dtype, m, n, compute_u=True, compute_v=True):
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
        """

        self.dtype = dtype.type if isinstance(dtype, np.dtype) else dtype
        self.dtypereal = self.dtype(0).real.dtype.type
        self.complex = np.iscomplexobj(self.dtype(0))

        from .lapack import _gesvd
        self.gesvd = _gesvd[self.dtype]

        self.m = m
        self.n = n
        mn = min(m, n)

        self.compute_u = compute_u
        self.compute_v = compute_v

        if not compute_u:
            self.jobu = "n"
            self.ldu = 0 # never referenced
            self.u_ptr = ctypes.c_void_p(0)
            self.write_u_into_a = False
        elif compute_v and m < n:
            self.jobu = "s"
            self.u = np.empty((m, mn), self.dtype, "F")
            self.ldu = self.u.strides[1] // self.u.itemsize
            self.u_ptr = self.u.ctypes.data_as(ctypes.c_void_p)
            self.write_u_into_a = False
        else:
            self.jobu = "o"
            self.ldu = m
            self.write_u_into_a = True

        if not compute_v:
            self.jobv = "n"
            self.ldv = 0 # never referenced
            self.v_ptr = ctypes.c_void_p(0)
            self.write_v_into_a = False
        elif compute_u and m >= n:
            self.jobv = "s"
            self.v = np.empty((mn, n), self.dtype, "F")
            self.ldv = self.v.strides[1] // self.v.itemsize
            self.v_ptr = self.v.ctypes.data_as(ctypes.c_void_p)
            self.write_v_into_a = False
        else:
            self.jobv = "o"
            self.ldv = mn
            self.write_v_into_a = True

        self.jobu = ctypes.byref(ctypes.c_char(self.jobu))
        self.jobv = ctypes.byref(ctypes.c_char(self.jobv))

        self.lda = m
        self.s = np.empty(mn, self.dtypereal)

        self.info = ctypes.c_int()

        work = np.empty(1, self.dtype)

        if self.complex:
            self.rwork = np.empty(5*mn, self.dtypereal)
            self.gesvd(
                self.jobu, self.jobv,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.ldu)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.ldv)),
                work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(-1)),
                self.rwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )
        else:
            self.gesvd(
                self.jobu, self.jobv,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.lda)),
                self.s.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.ldu)),
                ctypes.c_void_p(0),
                ctypes.byref(ctypes.c_int(self.ldv)),
                work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(-1)),
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

        a_ptr = a.ctypes.data_as(ctypes.c_void_p)

        if self.write_u_into_a:
            self.u = a[:,:min(self.m, self.n)]
            self.u_ptr = a_ptr

        if self.write_v_into_a:
            self.v = a[:min(self.m, self.n)]
            self.v_ptr = a_ptr

        if self.complex:
            self.gesvd(
                self.jobu, self.jobv,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                a_ptr, ctypes.byref(ctypes.c_int(self.lda)),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                self.work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lwork)),
                self.rwork.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(self.info)
            )
        else:
            self.gesvd(
                self.jobu, self.jobv,
                ctypes.byref(ctypes.c_int(self.m)),
                ctypes.byref(ctypes.c_int(self.n)),
                a_ptr, ctypes.byref(ctypes.c_int(self.lda)),
                self.s.ctypes.data_as(ctypes.c_void_p),
                self.u_ptr, ctypes.byref(ctypes.c_int(self.ldu)),
                self.v_ptr, ctypes.byref(ctypes.c_int(self.ldv)),
                self.work.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(ctypes.c_int(self.lwork)),
                ctypes.byref(self.info)
            )

        if self.compute_u:
            if self.compute_v:
                return self.u, self.s, self.v
            else:
                return self.u, self.s
        else:
            if self.compute_v:
                return self.s, self.v
            else:
                return self.s
