#!/usr/bin/env python
# coding: utf-8

# # Normal Equations vs Pseudoinverse

# In[3]:


import numpy as np
import numpy.linalg as la


# Here's a simple overdetermined linear system, which we'll solve using both the normal equations and the pseudoinverse:

# In[4]:


A = np.random.randn(5, 3)
b = np.random.randn(5)


# ### Normal Equations
# 
# Solve $Ax\cong b$ using the normal equations:

# In[5]:


x1 = la.solve(A.T@A, A.T@b)
x1


# ### Pseudoinverse
# 
# Solve $Ax\cong b$ using the pseudoinverse:

# In[6]:


U, sigma, VT = la.svd(A)
print(U)
print(sigma)
print(VT)


# In[7]:


Sigma_inv = np.zeros_like(A.T)
Sigma_inv[:3,:3] = np.diag(1/sigma)
Sigma_inv


# In[10]:


pinv = VT.T @ Sigma_inv @ U.T
x2 = pinv @ b
x2


# In[9]:


la.norm(x1-x2)


# In[ ]:




