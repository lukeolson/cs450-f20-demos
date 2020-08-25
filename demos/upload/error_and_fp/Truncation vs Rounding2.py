#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)


# In[3]:


hlist = np.logspace(0,-16,40)

plt.semilogx(hlist, 0*hlist, '|')


# In[4]:


x = 1
errors = []

for h in hlist:
    approxdf = (f(x+h) - f(x)) / h
    error = np.abs(df(x) - approxdf)
    errors.append(error)
    
errors = np.array(errors)


# In[5]:


plt.loglog(hlist, errors)
plt.xlabel('h')
plt.ylabel('error in the derivative')
plt.grid(True)


# In[6]:


plt.loglog(hlist, errors)
c = errors[0] / hlist[0]
plt.loglog(hlist, c * hlist , lw=3, zorder=-1)
plt.xlabel('h')
plt.ylabel('error in the derivative')
plt.grid(True)


# In[9]:


plt.loglog(hlist, errors)
c = errors[0] / hlist[0]
plt.loglog(hlist, c*hlist, lw=3, zorder=-1)

eps = 2**-53
plt.loglog(hlist, 2 * eps / hlist, lw=3, zorder=-2)

hopt = np.sqrt(2*eps/c)
plt.plot(hopt, np.abs(df(x) - (f(x+hopt)-f(x))/hopt), 'ms')
plt.xlabel('h')
plt.ylabel('error in the derivative')
plt.grid(True)


# In[ ]:


2**-52 / 2


# In[ ]:




