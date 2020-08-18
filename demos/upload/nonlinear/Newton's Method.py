#!/usr/bin/env python
# coding: utf-8

# # Newton's Method

# In[85]:


import numpy as np
import matplotlib.pyplot as pt


# Here's a function:

# In[86]:


def f(x):
    return x**3 - x +1

def df(x):
    return 3*x**2 - 1

xmesh = np.linspace(-2, 2, 100)
pt.ylim([-3, 10])
pt.plot(xmesh, f(xmesh))


# In[87]:


guesses = [2]


# Evaluate this cell many times in-place (using Ctrl-Enter)

# In[84]:


x = guesses[-1] # grab last guess

slope = df(x)

# plot approximate function
pt.plot(xmesh, f(xmesh))
pt.plot(xmesh, f(x) + slope*(xmesh-x))
pt.plot(x, f(x), "o")
pt.ylim([-3, 10])
pt.axhline(0, color="black")

# Compute approximate root
xnew = x - f(x) / slope
guesses.append(xnew)
print(xnew)

