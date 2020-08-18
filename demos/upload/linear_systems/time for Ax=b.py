#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


import timeit


# In[28]:


nlist = np.logspace(5, 13, base=2)
tlist = np.zeros(len(nlist))
for i, n in enumerate(nlist):
    n = int(n)
    A = np.random.rand(n,n)
    b = np.random.rand(n)
    t0 = timeit.default_timer()
    x = np.linalg.solve(A, b)
    t1 = timeit.default_timer()
    tlist[i] = t1-t0


# In[29]:


nlist = np.array(nlist)
plt.loglog(nlist, tlist)
plt.loglog(nlist, (nlist**3 / nlist[0]**3) * tlist[0])
plt.grid(True)


# In[ ]:




