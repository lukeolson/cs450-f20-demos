#!/usr/bin/env python
# coding: utf-8

# # Sherman-Morrison

# In[1]:


import numpy as np
import scipy.linalg as la


# Let's set up some matrices and data for the *rank-one modification*:

# In[6]:


n = 5
A = np.random.randn(n, n)
u = np.random.randn(n)
v = np.random.randn(n)

b = np.random.randn(n)

Ahat = A + np.outer(u, v)


# Let's start by computing the "base" factorization.
# 
# We'll use `lu_factor` from `scipy`, which stuffs both `L` and `U` into a single matrix (why can it do that?) and also returns pivoting information:

# In[7]:


LU, piv = la.lu_factor(A)
print(LU)
print(piv)


# Next, we set up a subroutine to solve using that factorization and check that it works:

# In[9]:


def solveA(b):
    return la.lu_solve((LU, piv), b)

la.norm(np.dot(A, solveA(b)) - b)


# As a last step, we try the Sherman-Morrison formula:
# 
# $$(A+uv^T)^{-1} = A^{-1} - {A^{-1}uv^T A^{-1} \over 1 + v^T A^{-1}u}$$

# To see that we got the right answer, we first compute the right solution of the modified system:

# In[11]:


xhat = la.solve(Ahat, b)


# Next, apply Sherman-Morrison to find `xhat2`:

# In[13]:


xhat2 = solveA(b) - solveA(u)*np.dot(v, solveA(b))/(1+np.dot(v, solveA(u)))


# In[14]:


la.norm(xhat - xhat2)


# * What's the cost of the Sherman-Morrison procedure?
