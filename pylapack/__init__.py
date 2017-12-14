# -*- coding: utf-8 -*-
"""
Created on 2017-12-12T09:17:10.557Z

@author: mrader1248
"""

########## python 2/3 compatibility ##########
from __future__ import absolute_import, division, print_function
import sys
if sys.version_info.major == 2:
    range = xrange
    xrange = None # to prevent accidental usage
##############################################


from . import lapack
from .lapack import init

from .gesvdx import GESVDX
