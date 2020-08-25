#!/usr/bin/env python
# coding: utf-8

# # Truncation Error vs Rounding Error

# In this notebook, we'll investigate two common sources of error: Truncation error and rounding error.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')


# **Task:** Approximate a function (here: a parabola, by a line)

# In[2]:


center = -1
width = 6

def f(x):
    return - x**2 + 3*x

def df(x):
    return -2*x + 3

grid = np.linspace(center-width/2, center+width/2, 100)

fx = f(grid)
plt.plot(grid, fx)
plt.plot(grid, f(center) + df(center) * (grid-center))

plt.xlim([grid[0], grid[-1]])
plt.ylim([np.min(fx), np.max(fx)])


# * What's the error we see?
# * What if we make `width` smaller?
