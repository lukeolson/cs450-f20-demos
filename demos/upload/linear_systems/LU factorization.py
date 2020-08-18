#!/usr/bin/env python
# coding: utf-8

# # LU Factorization

# In[2]:


import numpy as np
import numpy.linalg as la


# ## Part 1: One Column of LU

# In[3]:


n = 3

np.random.seed(15)
A = np.round(5*np.random.randn(n, n))

A


# Initialize `L` and `U` with zeros:

# In[4]:


L = np.zeros((n,n))
U = np.zeros((n,n))


# Set `U` to be the first row of `A`:

# In[5]:


U[0,:] = A[0,:]
U


# Compute the first column of `L`:

# In[6]:


L[:,0] = A[:,0]/U[0,0]
L


# Compare what we have to `A`:

# In[59]:


print(A)
print(L@U)


# Perform the Schur complement update and store the result in `A1`:

# In[61]:


A1 = A - L @ U
A1


# Take the second row of `U` to be the second row of `A1`:

# In[62]:


U[1,1:] = A1[1,1:]
U


# We can now compute the next column of `L`:

# In[64]:


L[1:,1] = A1[1:,1]/U[1,1]
L


# And finally, compute the bottom right elements of `L` and `U`

# In[65]:


U[2,2] = A1[2,2] - L[2,1]*U[1,2]
L[2,2] = 1.0


# In[66]:


print(L)
print(U)


# In[67]:


print(A)
print(L@U)


# ## Part 2: The Full Algorithm
# 
# Implement the general LU factorization algorithm

# In[7]:


n = 4
A = np.random.random((n,n)) 
L = np.zeros((n,n)) 
U = np.zeros((n,n)) 
M = A.copy()


# In[108]:


for i in range(n):
    U[i,i:] = M[i,i:]
    L[i:,i] = M[i:,i]/U[i,i]
    M[i+1:,i+1:] -= np.outer(L[i+1:,i:i+1],U[i:i+1,i+1:])   


# In[110]:


print(L)
print(U)
print(A-L@U)


# In[ ]:




