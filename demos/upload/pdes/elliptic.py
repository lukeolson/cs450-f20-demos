#!/usr/bin/env python
# coding: utf-8

# In[3]:


import scipy.sparse.linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import time
import numpy as  np
from structured_matrix import diff2d, diff3d


# In[28]:


n = 200
h = 1.0 / (n - 1)

A = diff2d(n, n)


# In[29]:


plt.spy(A, marker='s', ms=5)


# In[31]:


# exact solution u = sin(pi x) * sin(pi y)
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
f = h**2 * 2 * np.pi**2 * np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())
f =  0 *  f
f[2500] = 1.0

t = time.time()
u = la.spsolve(A, f)
t = time.time() - t
print("time = %g" % t)


# In[33]:


err = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel()) - u
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u.reshape(X.shape),
                rstride=1, cstride=1, cmap=plt.cm.jet, lw=0)
ax.axis('off')


# In[ ]:


Ainv = np.linalg.inv(A.toarray()) * h**2


# In[ ]:


np.linalg.norm(Ainv, np.inf)


# In[ ]:


h


# In[ ]:


1 / 8


# In[7]:


3
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))


# In[9]:


print(X)
print(Y)


# In[10]:


X.ravel()


# In[ ]:




