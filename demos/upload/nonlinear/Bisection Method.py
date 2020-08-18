#!/usr/bin/env python
# coding: utf-8

# # Bisection Method

# In[1]:


import numpy as np
import matplotlib.pyplot as pt


# In[2]:


a = 2
b = 6

x = np.linspace(a, b)

def f(x):
    return 1e-2 * np.exp(x) - 2

pt.grid()
pt.plot(x, f(x))


# Write code for the bisection method and run it in-place many times: (Ctrl-Enter)

# In[25]:


m = (a+b)/2

if np.sign(f(a)) == np.sign(f(m)):
    a = m
else:
    b = m
        
print(a, b)

