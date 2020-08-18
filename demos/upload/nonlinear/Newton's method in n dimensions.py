#!/usr/bin/env python
# coding: utf-8

# # Newton's method in $n$ dimensions

# In[2]:


import numpy as np
import numpy.linalg as la


# In[3]:


def f(xvec):
    x, y = xvec
    return np.array([
        x + 2*y -2,
        x**2 + 4*y**2 - 4
        ])


# In[4]:


def Jf(xvec):
    x, y = xvec
    return np.array([
        [1, 2],
        [2*x, 8*y]
        ])


# Pick an initial guess.

# In[5]:


x = np.array([1, 2])


# Now implement Newton's method.

# In[6]:


x = x - la.solve(Jf(x), f(x))
print(x)


# Check if that's really a solution:

# In[7]:


f(x)


# * What's the cost of one iteration?
# * Is there still something like quadratic convergence?

# --------------------
# Let's keep an error history and check.

# In[8]:


xtrue = np.array([0, 1])
errors = []
x = np.array([1, 2])


# In[19]:


x = x - la.solve(Jf(x), f(x))
errors.append(la.norm(x-xtrue))
print(x)


# In[20]:


for e in errors:
    print(e)


# In[21]:


for i in range(len(errors)-1):
    print(errors[i+1]/errors[i]**2)


# In[ ]:




