#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def psi(t, tpts):
    z = np.ones_like(t)
    for ti in tpts:
        z *= (t - ti)
    return z


# In[10]:


n = 5
tpts = np.linspace(0,1,n)
t = np.linspace(0,1,1000)

plt.plot(t, abs(psi(t, tpts)))


# In[7]:


n = 12
tpts = (np.cos((2*np.arange(1,n+1)-1)/(2*n) * np.pi) + 1) / 2
t = np.linspace(0,1,1000)

plt.plot(t, abs(psi(t, tpts)))


# In[ ]:




