#!/usr/bin/env python
# coding: utf-8

# # Gauss-Newton

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as pt
import scipy.linalg as la


# We would like to fit the model $f(t) = x_0 e^{x_1 t}$ to the following data using Gauss-Newton:

# In[2]:


t = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([2.0, 0.7, 0.3, 0.1])


# First, define a residual function (as a function of $\mathbf x=(x_0, x_1)$)

# In[3]:


def residual(x):
    return y - x[0] * np.exp(x[1] * t)


# Next, define its Jacobian matrix:

# In[11]:


def jacobian(x):
    return np.array([
        -np.exp(x[1] * t),
        -x[0] * t * np.exp(x[1] * t)
        ]).T


# In[12]:


jacobian(np.array([1, 0]))


# Here are two initial guesses. Try both:

# Here's a plotting function to judge the quality of our guess:

# In[6]:


def plot_guess(x):
    pt.plot(t, y, 'ro', markersize=20, clip_on=False)
    T = np.linspace(t.min(), t.max(), 100)
    Y = x[0] * np.exp(x[1] * T)
    pt.plot(T, Y, 'b-')
    
    print("Residual norm:", la.norm(residual(x), 2))

plot_guess(x)


# In[5]:


x = np.array([1, 0])
#x = np.array([0.4, 2])


# Code up one iteration of Gauss-Newton. Use `numpy.linalg.lstsq()` to solve the least-squares problem, noting that that function returns a tuple--the first entry of which is the desired solution.
# 
# Also print the residual norm. Use `plot_iterate` to visualize the current guess.
# 
# Then evaluate this cell in-place many times (Ctrl-Enter):

# In[7]:


#x = np.array([1, 0])
x = np.array([0.4, 2])
pt.figure()
plot_guess(x)
for i in range(20):
    pt.figure()

    x = x + la.lstsq(jacobian(x), -residual(x))[0]

    plot_guess(x)


# In[18]:


def residual(x):
    return y - (x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3)

def jacobian(x):
    return np.array([
        -t**0,
        -t,
        -t**2,
        -t**3
        ]).T

def plot_guess(x):
    pt.plot(t, y, 'ro', markersize=20, clip_on=False)
    T = np.linspace(t.min(), t.max(), 100)
    Y = x[0] + x[1]*T + x[2]*T**2 + x[3]*T**3
    pt.plot(T, Y, 'b-')
    
    print("Residual norm:", la.norm(residual(x), 2))


# In[19]:


x = np.array([1,1,1,1])
pt.figure()
plot_guess(x)
for i in range(3):
    pt.figure()

    x = x + la.lstsq(jacobian(x), -residual(x))[0]

    plot_guess(x)

