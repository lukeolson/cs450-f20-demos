#!/usr/bin/env python
# coding: utf-8

# # Chebyshev polynomials

# In[2]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


# ## Part I: Plotting the Chebyshev polynomials

# In[7]:


x = np.linspace(-1, 1, 100)

pt.xlim([-1.2, 1.2])
pt.ylim([-1.2, 1.2])

for k in range(5): # crank up
    pt.plot(x, np.cos(k*np.arccos(x)), lw=4)


# ## Part II: Understanding the Nodes
# 
# What if we interpolate random data?

# In[3]:


n = 20 # crank up


# ### "Extremal" Chebyshev Nodes (or: Chebyshev Nodes of the Second Kind)
# 
# * Most often used for computation
# * Note: Generates $n+1$ nodes -> drop $k$

# In[4]:


k = n-1

i = np.arange(0, k+1)
x = np.linspace(-1, 1, 3000)

def f(x):
    return np.cos(k*np.arccos(x))

nodes = np.cos(i/k*np.pi)

pt.plot(x, f(x))
pt.plot(nodes, f(nodes), "o")


# ### Chebyshev Nodes of the First Kind (Roots)
# 
# * Generates $n$ nodes

# In[5]:


i = np.arange(1, n+1)
x = np.linspace(-1, 1, 3000)

def f(x):
    return np.cos(n*np.arccos(x))

nodes = np.cos((2*i-1)/(2*n)*np.pi)

pt.plot(x, f(x))
pt.plot(nodes, f(nodes), "o")


# ### Observe Spacing

# In[6]:


pt.plot(nodes, 0*nodes, "o")


# ## Part III: Chebyshev Interpolation

# In[9]:


V = np.cos(i*np.arccos(nodes.reshape(-1, 1)))
data = np.random.randn(n)
coeffs = la.solve(V, data)


# In[10]:


x = np.linspace(-1, 1, 1000)
Vfull = np.cos(i*np.arccos(x.reshape(-1, 1)))
pt.plot(x, np.dot(Vfull, coeffs))
pt.plot(nodes, data, "o")


# ## Part IV: Conditioning

# In[13]:


n = 100 # crank up

i = np.arange(n, dtype=np.float64)
nodes = np.cos((2*(i+1)-1)/(2*n)*np.pi)
V = np.cos(i*np.arccos(nodes.reshape(-1, 1)))

la.cond(V)


# In[ ]:





# In[ ]:




