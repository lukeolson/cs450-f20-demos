#!/usr/bin/env python
# coding: utf-8

# # Convergence of Newton's Method

# In[4]:


import numpy as np
import matplotlib.pyplot as pt


# In[5]:


def f(x):
    return np.exp(x) - 2

def df(x):
    return np.exp(x)


# In[8]:


xgrid = np.linspace(-2, 3, 1000)
pt.grid()
pt.plot(xgrid, f(xgrid))


# What's the true solution of $f(x)=0$?

# In[9]:


xtrue = np.log(2)
print(xtrue)
print(f(xtrue))


# Now let's run Newton's method and keep track of the errors:

# In[18]:


errors = []
x = 2


# At each iteration, print the current guess and the error.

# In[24]:


x = x - f(x)/df(x)
print(x)
errors.append(abs(x-xtrue))
print(errors[-1])


# In[25]:


for err in errors:
    print(err)


# * Do you have a hypothesis about the order of convergence?

# In[27]:


# Doubles number of digits each iteration: probably quadratic.


# ------------
# Let's check:

# In[26]:


for i in range(len(errors)-1):
    print(errors[i+1]/errors[i]**2)


# In[ ]:




