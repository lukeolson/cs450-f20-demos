#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import scipy.linalg as sla
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from time import time


# In[6]:


nlist = np.logspace(1,3,20)

times = []

for n in nlist:
    n = int(n)
    A = np.random.rand(n,n)
    B = np.random.rand(n)
    
    t0 = time()
    sla.solve(A, B)
    t1 = time() - t0
    times.append(t1)
    print(n)


# In[7]:


plt.loglog(nlist, times)
plt.grid(True)
plt.xlabel("n")
plt.ylabel('time (s)')


# In[ ]:




