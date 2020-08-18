#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def f(x):
    return x - x**6


# In[3]:


x = np.linspace(0,1,100)
plt.plot(x, f(x))


# In[4]:


z = np.sort(np.random.rand(10))
z[0] = 0
z[-1] = 1
h = np.diff(z)
Ih = np.inner(h, f(z[:-1]))
print(Ih)


# In[5]:


plt.plot(x, f(x))
plt.plot(z, 0*z, 'o', ms=12)


# In[6]:


I = 1/2 - 1/7
print(I)


# In[7]:


for i in range(20):
    znew = np.zeros(len(z)*2 - 1)
    znew[0::2] = z
    znew[1::2] = (z[:-1] + z[1:]) / 2
    z = znew
    h = np.diff(z)
    Ih = np.inner(h, f(z[:-1]))
    print(I - Ih)


# In[ ]:


len(z)


# In[ ]:


len(znew[2::2])


# In[ ]:




