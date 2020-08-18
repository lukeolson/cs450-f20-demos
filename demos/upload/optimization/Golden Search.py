#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_context('talk')


# In[2]:


def f(x):
    return x**2


# In[3]:


a = -3
b = 4
e = []


# In[35]:



xx = np.linspace(-4,4,20)

plt.plot(xx, f(xx), 'k-')
plt.plot(a, f(a), 'ro', b, f(b), 'ro')

x1 = a + (1-(np.sqrt(5)-1)/2) * (b-a)
x2 = a + (np.sqrt(5)-1)/2 * (b-a)

if f(x1) > f(x2):
    a = x1
else:
    b = x2
    
e.append((a+b)/2)
print("{}".format(e[-1]))


# In[36]:


plt.semilogy(np.abs(e))


# In[ ]:




