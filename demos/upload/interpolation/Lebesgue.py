#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def lagrange(i, x, xx):
    L = np.ones(len(xx))
    
    for j in range(len(x)):
        if j != i:
            L *= (xx - x[j]) / (x[i] - x[j])    
    return L

def lebesgue(x, xx):
    L = np.zeros(len(xx))
    for i in range(len(x)):
        yy = lagrange(i, x, xx)
        L += np.abs(yy)
        
    return L


# In[8]:


xx = np.linspace(-1,1,10000)
x = np.linspace(-1,1,12)
#k = 15
#x = np.cos((2*np.arange(1,k+1)-1)/(2*k) * np.pi)
plt.plot(x, 0*x, 'ko')
plt.plot(x, 1+0*x, 'ko')

for i in range(len(x)):
    yy = lagrange(i, x, xx)
    plt.plot(xx, yy)


# In[9]:


L = lebesgue(x, xx)

plt.plot(xx, L)


# In[ ]:





# In[ ]:




