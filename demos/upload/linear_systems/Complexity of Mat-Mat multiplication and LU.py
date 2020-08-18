#!/usr/bin/env python
# coding: utf-8

# # Relative cost of matrix operations

# In[1]:


import numpy as np
import scipy.linalg as spla
import scipy as sp

import matplotlib.pyplot as pt

from time import time

np.alterdot()


# In[2]:


n_values = (10**np.linspace(1, 3.25, 15)).astype(np.int32)
n_values


# In[3]:



def mat_mul(A):
    return A.dot(A)

for name, f in [
        ("mat_mul", mat_mul),
        ("lu", spla.lu_factor),
        ]:

    times = []
    print("----->", name)
    
    for n in n_values:
        print(n)

        A = np.random.randn(n, n)
        
        start_time = time()
        f(A)
        times.append(time() - start_time)
        
    pt.plot(n_values, times, label=name)

pt.grid()
pt.legend(loc="best")
pt.xlabel("Matrix size $n$")
pt.ylabel("Wall time [s]")


# * The faster algorithms make the slower ones look bad. But... it's all relative.
# * Is there a better way of plotting this?
# * Can we see the asymptotic cost ($O(n^3)$) of these algorithms from the plot?

# In[3]:




