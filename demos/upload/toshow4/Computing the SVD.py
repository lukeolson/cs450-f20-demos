#!/usr/bin/env python
# coding: utf-8

# # Computing the SVD

# In[1]:


import numpy as np
import numpy.linalg as la


# In[2]:


np.random.seed(15)
n = 5
A = np.random.randn(n, n)


# Now compute the eigenvalues and eigenvectors of $A^TA$ as `eigvals` and `eigvecs` using `la.eig` or `la.eigh` (symmetric):

# In[11]:


eigvals, eigvecs = la.eigh(A.T @ A)


# In[12]:


eigvals


# Eigenvalues are real and positive. Coincidence?

# In[13]:


eigvecs.shape


# Check that those are in fact eigenvectors and eigenvalues:

# In[14]:


B = A.T @ A
B - eigvecs @ np.diag(eigvals) @ la.inv(eigvecs)


# `eigvecs` are orthonormal! (Why?)
# 
# Check:

# In[15]:


la.norm(eigvecs.T @ eigvecs  - np.eye(n))


# ------
# Now piece together the SVD:

# In[16]:


Sigma = np.diag(np.sqrt(eigvals))


# In[17]:


V = eigvecs


# In[18]:


U = A @ V @ la.inv(Sigma)


# Check orthogonality of `U`:

# In[19]:


U @ U.T - np.eye(n)


# In[ ]:




