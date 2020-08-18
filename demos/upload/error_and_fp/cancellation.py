#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


for p in np.arange(0,10):
    x = 10**p
    f1 = x * (np.sqrt(x+1) - np.sqrt(x))
    f2 = x / (np.sqrt(x+1) + np.sqrt(x))
    print(f"{x:>10} {f1:>20} {f2:>20} {f1-f2:>20}")


# In[7]:


### x=np.linspace(0.99998, 1.00002, 100)

#f = (x-1)**3
#f = x**3 - 3*x**2 + 3*x - 1
f = -1+x*(3+x*(-3+x))
plt.plot(x, f, 'o')


# In[ ]:




