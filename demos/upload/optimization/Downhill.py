#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_ncg

import seaborn as sns
sns.set_context('talk')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm

from mpl_toolkits import mplot3d

get_ipython().run_line_magic('matplotlib', 'notebook')


# # Which way is "down"?

# ### bowl
# 
# Let's look at a standard "bowl" given by
# $$
# f(x,y) = x^2 +  y^2
# $$
# which has a minimum at  $(x,y)=(0,0)$

# In[3]:


def f(x,y):
    return x**2 + y**2

def df(x,y):
    return np.array([2*x,2*y])


# In[4]:


x0 = np.arange(1,1)

fig = plt.figure()
ax = Axes3D(fig, azim = -80, elev =55)

X = np.linspace(-2, 2, 30)
Y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = 'viridis')

ax.plot([0], [0], [f(0,0)], 'mo', ms=10, zorder=10)


# In[5]:


plt.figure()
plt.contourf(X, Y, Z, levels=30, cmap = 'cool')
xx = 1.5
yy = 1.5
s = -0.5*df(xx, yy)
plt.plot(xx, yy, 'bo', ms=10, zorder=10)
arrow = plt.Arrow(xx, yy, s[0], s[1], zorder=10, width=0.5)
ax = plt.gca()
ax.add_patch(arrow)


# ### bread bowl
# 
# Let's look at a bread bowl given by
# $$
# f(x,y) = 10 * x^2 +  y^2
# $$
# which has a minimum at  $(x,y)=(0,0)$

# In[6]:


def f(x,y):
    return 10 * x**2 + y**2

def df(x,y):
    return np.array([20*x,2*y])


# In[7]:


x0 = np.arange(1,1)

fig = plt.figure()
ax = Axes3D(fig, azim = -80, elev =55)

X = np.linspace(-2, 2, 30)
Y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = 'viridis')

ax.plot([0], [0], [f(0,0)], 'mo', ms=10, zorder=10)


# In[8]:


plt.figure()
plt.contourf(X, Y, Z, levels=30, cmap = 'cool')
xx = 1.5
yy = 1.5
s = -0.1*df(xx, yy)
plt.plot(xx, yy, 'bo', ms=10, zorder=10)
arrow = plt.Arrow(xx, yy, s[0], s[1], zorder=10, width=0.5)
ax = plt.gca()
ax.add_patch(arrow)


# ### Rosenbrock
# 
# Let's look at a Rosenbrock function
# $$
# f(x,y) = (a-x)^2 + b(y-x^2)^2
# $$
# which has a minimum at  $(x,y)=(a,a^2)$
# 
# https://en.wikipedia.org/wiki/Rosenbrock_function

# In[12]:


def f(x,y):
    a = 1
    b = 100
    return (a - x)**2 + b*(y - x**2)**2


# In[13]:


x0 = np.arange(1,1)

fig = plt.figure()
ax = Axes3D(fig, azim = -80, elev =55)

X = np.linspace(-4, 4, 30)
Y = np.linspace(-4, 4, 30)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, norm = LogNorm(), cmap = 'viridis')

ax.plot([1], [1], [f(1,1)], 'mo', ms=10, zorder=10)
ax.plot([2], [2], [f(2,2)], 'mo', ms=10, zorder=10)
ax.plot([1.8], [3.2], [f(1.8,3.2)], 'mo', ms=10, zorder=10)


# In[11]:


plt.figure()
plt.contourf(X, Y, Z, levels=30, cmap = 'cool')
xx = 1.8
yy = 3.2
s = -0.1*df(xx, yy)
plt.plot(xx, yy, 'bo', ms=10, zorder=10)
plt.plot(1, 1, 'm*', ms=10, zorder=10)
arrow = plt.Arrow(xx, yy, s[0], s[1], zorder=10, width=0.5)
ax = plt.gca()
ax.add_patch(arrow)


# In[ ]:


def f(x,y):
    a = 1
    b = 0.5
    return (a - x)**2 + b*(y - x**2)**2

