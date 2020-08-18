#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[18]:


n = 100
A = np.random.rand(n,n)


# In[19]:


np.linalg.cond(A)


# In[20]:


x = np.ones(n)
b = A.dot(x)


# In[21]:


xhat = np.linalg.solve(A, b)


# In[22]:


np.linalg.norm(x-xhat) / np.linalg.norm(x)


# In[ ]:




