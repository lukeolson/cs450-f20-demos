#!/usr/bin/env python
# coding: utf-8

# # Arnoldi Iteration

# In[2]:


import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as pt


# Let us make a matrix with a defined set of eigenvalues and eigenvectors, given by `eigvals` and `eigvecs`.

# In[3]:


np.random.seed(40)

# Generate matrix with eigenvalues 1...25
n = 25
eigvals = np.linspace(1., n, n)
eigvecs = np.random.randn(n, n)
print(eigvals)

A = la.solve(eigvecs, np.dot(np.diag(eigvals), eigvecs))
print(la.eig(A)[0])


# ## Initialization

# Set up $Q$ and $H$:

# In[4]:


Q = np.zeros((n, n))
H = np.zeros((n, n))

k = 0


# Pick a starting vector, normalize it

# In[5]:


x0 = np.random.randn(n)
x0 = x0/la.norm(x0)

# Poke it into the first column of Q
Q[:, k] = x0

del x0


# Make a list to save arrays of Ritz values:

# In[6]:


ritz_values = []


# ## Algorithm

# Carry out one iteration of Arnoldi iteration.
# 
# Run this cell in-place (Ctrl-Enter) until H is filled.

# In[32]:


print(k)

u = A @ Q[:, k]

# Carry out Gram-Schmidt on u against Q
for j in range(k+1):
    qj = Q[:, j]
    H[j,k] = qj @ u
    u = u - H[j,k]*qj

if k+1 < n:
    H[k+1, k] = la.norm(u)
    Q[:, k+1] = u/H[k+1, k]

k += 1

pt.spy(H)

ritz_values.append(la.eig(H)[0])


# Check that $Q^T A Q =H$:

# In[33]:


la.norm(Q.T @ A @ Q - H)/ la.norm(A)


# Check that Q is orthogonal:

# In[34]:


la.norm(Q.T @ Q - np.eye(n))


# ## Plot convergence of Ritz values

# Enable the Ritz value collection above to make this work.

# In[37]:


pt.figure(figsize=(10,10))
for i, rv in enumerate(ritz_values):
    I = np.where(rv != 0.0)[0]
    pt.plot([i] * len(rv[I]), rv[I], "x")


# In[ ]:


A = np.random.rand(2,2) + 1j*np.random.rand(2,2)
print(A)


# In[ ]:


A.conj()


# In[ ]:


A.conj().T


# In[ ]:


import scipy.linalg as sla


# In[ ]:


sla.


# In[36]:


ritz_values


# In[ ]:




