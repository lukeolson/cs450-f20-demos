#!/usr/bin/env python
# coding: utf-8

# # Gram-Schmidt and Modified Gram-Schmidt

# In[1]:


import numpy as np
import numpy.linalg as la


# In[2]:


A = np.random.randn(3, 3)


# In[11]:


def test_orthogonality(Q):
    print("Q:")
    print(Q)
    
    print("Q^T Q:")
    QtQ = np.dot(Q.T, Q)
    QtQ[np.abs(QtQ) < 1e-15] = 0
    print(QtQ)


# In[12]:


Q = np.zeros(A.shape)


# Now let us generalize the process we used for three vectors earlier:

# In[13]:


for k in range(A.shape[1]):
    avec = A[:, k]
    q = avec
    for j in range(k):
        q = q - np.dot(avec, Q[:,j])*Q[:,j]
    
    Q[:, k] = q/la.norm(q)


# This procedure is called [Gram-Schmidt Orthonormalization](https://en.wikipedia.org/wiki/Gram–Schmidt_process).

# In[14]:


test_orthogonality(Q)


# Now let us try a different example ([Source](http://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/)):

# In[15]:



np.set_printoptions(precision=13)

eps = 1e-8

A = np.array([
    [1,  1,  1],
    [eps,eps,0],
    [eps,0,  eps]
    ])

A


# In[16]:


Q = np.zeros(A.shape)


# In[17]:


for k in range(A.shape[1]):
    avec = A[:, k]
    q = avec
    for j in range(k):
        print(q)
        q = q - np.dot(avec, Q[:,j])*Q[:,j]
    
    print(q)
    q = q/la.norm(q)
    Q[:, k] = q
    print("norm -->", q)
    print("-------")


# In[18]:


test_orthogonality(Q)


# Questions:
# 
# * What happened?
# * How do we fix it?

# In[ ]:


Q = np.zeros(A.shape)


# In[19]:


for k in range(A.shape[1]):
    q = A[:, k]
    for j in range(k):
        q = q - np.dot(q, Q[:,j])*Q[:,j]
    
    Q[:, k] = q/la.norm(q)


# In[21]:


test_orthogonality(Q)


# This procedure is called *Modified* Gram-Schmidt Orthogonalization.
# 
# Questions:
# 
# * Is there a difference mathematically between modified and unmodified?
# * Why are there $10^{-8}$ values left in $Q^TQ$?

# In[ ]:




