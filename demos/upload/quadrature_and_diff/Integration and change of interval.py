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
    return x**2 * np.sin(x)**3


# In[4]:


xx = np.linspace(0,6,100)
plt.plot(xx, f(xx), 'b-', lw=4)

xx = np.linspace(2,5,100)
plt.fill_between(xx, f(xx), alpha=0.5)


# In[5]:


tn, wn = np.polynomial.legendre.leggauss(4)


# In[8]:


In = ((3/2) * f((3*tn + 7)/2) * wn).sum()
print(In)


# In[7]:


plt.plot(tn, tn*0, 'o')


# In[13]:


n = 5

def integrate(f, a, b, n):
    tn, wn = np.polynomial.legendre.leggauss(n)
    
    g = (b-a)/2 * f((b-a)*(tn+1)/2 + a)
    g *= wn
    return g.sum()

def testing(n):
    def f(x):
        # return 15*np.ones(len(x))
        return 0*x + 15
    
    np.testing.assert_almost_equal(integrate(f, 8, 10, 1), 30)
    
testing(n)


# In[16]:


integrate(f, 2, 5, 3)


# In[17]:


for n in range(1,15):
    In = integrate(f, 2, 5, n)
    print(In)


# In[18]:


import scipy


# In[20]:


scipy.integrate.quadrature(f, 2, 5)


# In[21]:


get_ipython().run_line_magic('pinfo', 'scipy.integrate')


# In[ ]:




