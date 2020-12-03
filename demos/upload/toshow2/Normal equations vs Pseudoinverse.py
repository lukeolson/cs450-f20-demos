#!/usr/bin/env python
# coding: utf-8

# # Normal Equations vs Pseudoinverse

# In[1]:


import numpy as np
import numpy.linalg as la


# Here's a simple overdetermined linear system, which we'll solve using both the normal equations and the pseudoinverse:

# In[2]:


A = np.random.randn(5, 3)
b = np.random.randn(5)


# ### Normal Equations
# 
# Solve $Ax\cong b$ using the normal equations:

# In[3]:


x1 = la.solve(A.T@A, A.T@b)
x1


# ### Pseudoinverse
# 
# Solve $Ax\cong b$ using the pseudoinverse:

# In[4]:


U, sigma, VT = la.svd(A)
print(U)
print(sigma)
print(VT)


# In[5]:


Sigma_inv = np.zeros_like(A.T)
Sigma_inv[:3,:3] = np.diag(1/sigma)
Sigma_inv


# In[6]:


pinv = VT.T @ Sigma_inv @ U.T
x2 = pinv @ b
x2


# In[7]:


la.norm(x1-x2)


# In[ ]:




