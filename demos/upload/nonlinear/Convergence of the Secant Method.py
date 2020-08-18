#!/usr/bin/env python
# coding: utf-8

# # Convergence of Newton's Method

# In[1]:


import numpy as np
import matplotlib.pyplot as pt


# In[2]:


def f(x):
    return np.exp(x) - 2


# In[3]:


xgrid = np.linspace(-2, 3, 1000)
pt.grid()
pt.plot(xgrid, f(xgrid))


# What's the true solution of $f(x)=0$?

# In[4]:


xtrue = np.log(2)
print(xtrue)
print(f(xtrue))


# Now let's run Newton's method and keep track of the errors:

# In[5]:


errors = []
x = 2
xbefore = 3


# At each iteration, print the current guess and the error.

# In[17]:


slope = (f(x)-f(xbefore))/(x-xbefore)

xbefore = x
x = x - f(x)/slope
print(x)
errors.append(abs(x-xtrue))
print(errors[-1])


# In[18]:


for err in errors:
    print(err)


# * Do you have a hypothesis about the order of convergence?

# In[19]:


# Does not quite double the number of digits each round--unclear.


# ------------
# Let's check:

# In[19]:


for i in range(len(errors)-1):
    print(errors[i+1]/errors[i]**1.618)


# In[ ]:




