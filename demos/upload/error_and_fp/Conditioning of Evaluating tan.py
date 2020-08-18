#!/usr/bin/env python
# coding: utf-8

# # Conditioning of evaluating tan()

# In[2]:


import numpy as np
import matplotlib.pyplot as pt


# Let us estimate the sensitivity of evaluating the $\tan$ function:

# In[3]:


x = np.linspace(-5, 5, 1000)
pt.ylim([-10, 10])
pt.plot(x, np.tan(x))


# In[18]:


x = np.pi/2 - 0.0001
#x = 0.1
x


# In[19]:


np.tan(x)


# In[20]:


dx = 0.00005
np.tan(x+dx)


# ## Condition number estimates

# ### From evaluation data
# 

# In[21]:



np.abs(np.tan(x+dx) - np.tan(x))/np.abs(np.tan(x)) / (np.abs(dx) / np.abs(x))


# ### Using the derivative estimate

# In[22]:


import sympy as sp

xsym = sp.Symbol("x")

f = sp.tan(xsym)
df = f.diff(xsym)
df


# Evaluate the derivative estimate. Use `.subs(xsym, x)` to substitute in the value of `x`.

# In[23]:


(xsym*df/f).subs(xsym, x)


# In[ ]:




