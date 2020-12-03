#!/usr/bin/env python
# coding: utf-8

# # Keeping track of coefficients in Gram-Schmidt

# In[2]:


import numpy as np
import numpy.linalg as la


# In[3]:


A = np.random.randn(3, 3)


# Let's start from regular old (modified) Gram-Schmidt:

# In[4]:



Q = np.zeros(A.shape)

q = A[:, 0]
Q[:, 0] = q/la.norm(q)

# -----------

q = A[:, 1]
coeff = np.dot(Q[:, 0], q)
q = q - coeff*Q[:, 0]
Q[:, 1] = q/la.norm(q)

# -----------

q = A[:, 2]
coeff = np.dot(Q[:, 0], q)
q = q - coeff*Q[:, 0]
coeff = np.dot(Q[:, 1], q)
q = q - coeff*Q[:, 1]
Q[:, 2] = q/la.norm(q)


# In[5]:


Q.dot(Q.T)


# Now we want to keep track of what vector got added to what other vector, in the style of an elimination matrix.
# 
# Let's call that matrix $R$.
# 
# * Would it be $A=QR$ or $A=RQ$? Why?
# * Where are $R$'s nonzeros?

# In[6]:


R = np.zeros((A.shape[0], A.shape[0]))


# In[7]:


Q = np.zeros(A.shape)

q = A[:, 0]
Q[:, 0] = q/la.norm(q)

R[0,0] = la.norm(q)

# -----------

q = A[:, 1]
coeff = np.dot(Q[:, 0], q)
R[0,1] = coeff
q = q - coeff*Q[:, 0]
Q[:, 1] = q/la.norm(q)

R[1,1] = la.norm(q)

# -----------

q = A[:, 2]
coeff = np.dot(Q[:, 0], q)
R[0,2] = coeff
q = q - coeff*Q[:, 0]
coeff = np.dot(Q[:, 1], q)
R[1,2] = coeff
q = q- coeff*Q[:, 1]
Q[:, 2] = q/la.norm(q)

R[2,2] = la.norm(q)


# In[8]:


R


# In[9]:


la.norm(Q@R - A)


# This is called [QR factorization](https://en.wikipedia.org/wiki/QR_decomposition).

# ----------
# * When does it break?
# * Does it need something like pivoting?
# * Can we use it for something?
