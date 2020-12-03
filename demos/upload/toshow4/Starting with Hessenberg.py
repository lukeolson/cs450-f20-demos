#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# Start with matrix $A$.  What happens to $A_k$ in each iteration?
# 1. If $A$ is standard full matrix, then each $A_k$ remains full
# 2. If $A$ is upper triangular, then $A_k$ will stay upper triangular
# 3. If $A$ is upper Hessenberg, then $A_k$ will stay upper Hessenberg
# 
# How do we transform $A$ into an upper Hessenberg matrix?

# In[8]:


A = np.random.rand(5,5)
case = 3
if case == 2:
    A = np.triu(A, k=0)
if case == 3:
    A = np.triu(A, k=-1)


# In[9]:


Q, R = np.linalg.qr(A)
A1 = R @ Q
Q, R = np.linalg.qr(A1)
A2 = R @ Q

with np.printoptions(precision=2):
    print(A)
    print(A1)
    print(A2)


# In[11]:


with np.printoptions(precision=2):
    print(Q)


# In[ ]:


R


# In[ ]:




