#!/usr/bin/env python
# coding: utf-8

# # Elimination Matrices I: The Basics

# In[1]:


import numpy as np


# In[2]:


n = 4


# ----------------
# Let's create an elimination matrix as $M$:

# In[24]:


M = np.eye(n)
M[1,0] = 2
M


# Here's a matrix $A$. See if $M$ has the desired effect on $A$:

# In[19]:


np.random.seed(5)
A = np.random.randn(n, n).round(1)
A


# In[20]:


M.dot(A)


# -----------------------
# Next, see if you can build the inverse of $M$:

# In[25]:


Minv = np.eye(n)
Minv[1,0] = -2
Minv


# In[26]:


M.dot(Minv)

