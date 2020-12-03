#!/usr/bin/env python
# coding: utf-8

# # Issues with the normal equations

# In[1]:


import numpy as np
import numpy.linalg as la


# Here's an example matrix to use with the normal equations:

# In[4]:


eps = 1e-10  # set to 1e-5, 1e-10

A = np.array([
        [1, 1],
        [eps, 0],
        [0, eps],
        ])

np.set_printoptions(precision=20)
print(A)
print(A.T @ A)


# * What do you notice about the entries of $A^T A$?

# In[6]:


n = 5

A = np.random.randn(5, 5) * 10**-np.linspace(0, -5, n)
print(A)
print(la.cond(A))


# In[7]:


la.cond(np.dot(A.T, A))


# * What do you notice about the condition number?
# * What's a general bound? $\operatorname{cond}(AB)\le \dots$?

# In[8]:


78701.17330588339**2


# In[ ]:




