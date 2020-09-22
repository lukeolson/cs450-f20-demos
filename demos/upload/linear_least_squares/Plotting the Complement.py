#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


seed = 34
np.random.seed(seed)
A = np.random.rand(3,2)
I = np.eye(3)
P = A @ np.linalg.inv(A.T @ A) @ A.T

x = np.random.rand(3,1000)
y = P @ x
w = (I - P) @ x

mmax = np.amax(np.maximum(y, w), axis=1)
mmin = np.amin(np.minimum(y, w), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[0,:], y[1,:], y[2,:], c='r')
ax.scatter(w[0,:], w[1,:], w[2,:], c='b')
ax.set_xlabel('x')
ax.set_ylabel('y')


# In[26]:


np.random.seed(seed)
A = np.random.rand(3,2)

U, s, VT = np.linalg.svd(A)

x = np.random.rand(2,1000)
y = U[:,:2] @ x

w = U[:,-1].reshape((3,1)) @ np.random.rand(1000).reshape((1,1000))

mmax = np.amax(np.maximum(y, w), axis=1)
mmin = np.amin(np.minimum(y, w), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y[0,:], y[1,:], y[2,:], c='r')
ax.scatter(w[0,:], w[1,:], w[2,:], c='b')
ax.set_xlabel('x')
ax.set_ylabel('y')

