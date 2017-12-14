# pylapack
LAPACK bindings for Python

This package provides bindings to LAPACK functions making it easy to reuse working memory.

The following example shows how `pylapack` can be used to perform a low-rank SVD using the ZGESVDX function (which cannot be used from `scipy`).

```python
import numpy as np
import scipy.linalg as la
from time import time

import pylapack
pylapack.init()

n = 4096
k = 256
a = np.random.rand(n,n) + 1j*np.random.rand(n,n)
a = np.asfortranarray(a)

t = time()
u2,s2,v2 = la.svd(a)
u2,s2,v2 = u2[:,:k],s2[:k],v2[:k]
t = time() - t
print "scipy:    {:f} s".format(t)

t = time()
svd = pylapack.GESVDX(a.dtype, n, n, svals=(0, k-1))
u,s,v = svd.run(a)
t = time() - t
print "pylapack: {:f} s".format(t)

assert np.allclose(s, s2)
```

Output on an Intel® Core™ i7-5960X:
```
scipy:    28.297044 s
pylapack: 21.041771 s
```
