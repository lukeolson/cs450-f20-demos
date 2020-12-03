#!/usr/bin/env python
# coding: utf-8

# # 3x3 Householder QR Demo

# This demo constructs a $3\times 3$ QR factorization using Householder reflectors.

# In[11]:


import numpy as np
import numpy.linalg as la


# In[12]:


n = 3

e1 = np.array([1,0,0])
e2 = np.array([0,1,0])
e3 = np.array([0,0,1])

A = np.random.randn(n, n)
A


# Householder reflector:
# $$I-2\frac{vv^T}{v^Tv}$$
# 
# Choose $v=a-\|a\|e_1$.

# In[13]:


a = A[:, 0]
v = a-la.norm(a)*e1

H1 = np.eye(3) - 2*np.outer(v, v)/(v@v)


# In[14]:


A1 = H1 @ A
A1


# NB: Never build full Householder matrices in actual code! (Why? How?)

# In[17]:


a = A1[:, 1].copy()
a[0] = 0
v = a-la.norm(a)*e2

H2 = np.eye(3) - 2*np.outer(v, v)/(v@v)


# In[18]:


R = H2 @ A1
R


# In[9]:


Q = np.dot(H2, H1).T
la.norm(np.dot(Q, R) - A)


# In[10]:


Q.T @ Q


# In[ ]:




