#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_context('talk')

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


# Examples are taken from here: http://terpconnect.umd.edu/~petersd/666/fixedpoint.pdf

# # Test Case 1
# 
# Let's start with a function
# $$
# \begin{align}
# x_0 & = 0.1 * (1 - x_1 - np.sin(x_0 + x_1))\\
# x_1 & = 0.1 * (2 + x_0 + np.cos(x_0 - x_1))
# \end{align}
# $$

# In[2]:


def g(x):
    xnew = np.zeros_like(x)
    xnew[0] = 0.1 * (1 - x[1] - np.sin(x[0] + x[1]))
    xnew[1] = 0.1 * (2 + x[0] + np.cos(x[0] - x[1]))
    return xnew


# In[3]:


x = np.zeros(2)

for i in range(10):
    x = g(x)
    print(x)


# In[4]:


x = np.zeros(2)
q = 0.3

for i in range(10):
    xnew = g(x)
    
    dk = q / (1 - q) * np.linalg.norm(xnew - x, np.inf)
    
    x = xnew
    print(x, dk)


# In[5]:


x = np.zeros(2)
q = 0.3

plt.figure(figsize=(8,8))
plt.plot(x[0], x[1], 'ro')
plt.axis([-1,1,-1,1])
dks = []

for i in range(10):
    xnew = g(x)
    
    dk = q / (1 - q) * np.linalg.norm(xnew - x, np.inf)
    dks.append(dk)
    
    x = xnew
    print(x, dk)
    
    # plot
    plt.plot(x[0], x[1], 'b.')
    ax = plt.gca()
    ax.add_patch(Rectangle((x[0]-dk, x[1]-dk), 2*dk, 2*dk, fill=None, alpha=1))


# In[6]:


plt.semilogy(dks)


# # Test Case 2
# 
# For this problem try
# $$
# \begin{align}
# x_0 = (1/3) * (x_0 - x_0*x_1 + 1)\\
# x_1 = (1/3) * (x_1 + x_0*x_1**2 + 1)
# \end{align}
# $$

# In[ ]:


def g(x):
    xnew = np.zeros_like(x)
    xnew[0] = (1/3) * (x[0] - x[0]*x[1] + 1)
    xnew[1] = (1/3) * (x[1] + x[0]*x[1]**2 + 1)
    return xnew


# In[7]:


x = np.zeros(2)
q = 19 / 27

for i in range(10):
    xnew = g(x)
    
    dk = q / (1 - q) * np.linalg.norm(xnew - x, np.inf)
    
    x = xnew
    print(x, dk)


# In[8]:


x = np.zeros(2)
q = 19/27

plt.figure(figsize=(8,8))
plt.plot(x[0], x[1], 'ro')
plt.axis([-1,1,-1,1])
plt.plot(x[0], x[1], 'b.')
ax = plt.gca()
ax.add_patch(Rectangle((-2/3,-2/3), 2*2/3, 2*2/3, fill=None, alpha=1,
                       edgecolor='r', linestyle='--'))
    
for i in range(10):
    xnew = g(x)
    
    dk = q / (1 - q) * np.linalg.norm(xnew - x, np.inf)
    
    x = xnew
    print(x, dk)
    
    # plot
    plt.plot(x[0], x[1], 'b.')
    ax = plt.gca()
    ax.add_patch(Rectangle((x[0]-dk, x[1]-dk), 2*dk, 2*dk, fill=None, alpha=1))


# In[ ]:


get_ipython().run_line_magic('pinfo', 'Rectangle')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




