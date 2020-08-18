#!/usr/bin/env python
# coding: utf-8

# # Coding Back-Substitution

# In[1]:


import numpy as np


# Here's an upper-triangular matrix $A$ and two vectors $x$ and $b$ so that $Ax=b$.
# 
# See if you can find $x$ by computation.

# In[2]:


n = 5

A = np.random.randn(n, n) * np.tri(n).T
print(A)

x = np.random.randn(n)
print(x)

b = A @ x


# In[3]:


xcomp = np.zeros(n)

for i in range(n-1, -1, -1):
    tmp = b[i]
    for j in range(n-1, i, -1):
        tmp -= xcomp[j]*A[i,j]
        
    xcomp[i] = tmp/A[i,i]


# Now compare the computed $x$ against the reference solution.

# In[4]:


print(x)
print(xcomp)


# Questions/comments:
# 
# * Can this fail?
# * What's the operation count?
