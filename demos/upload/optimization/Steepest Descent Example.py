#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk')
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.optimize as sopt


# In[65]:


def f(x):
    return 5*x[0]**2 + x[1]**2 + 4*x[0]*x[1] - 14*x[0] - 6*x[1] + 20

def df(x):
    s = np.zeros((2,))
    s[0] = 10*x[0] + 4*x[1] - 14
    s[1] = 2*x[1] + 4*x[0] - 6
    return s
xstar = np.array([1,1])


# In[67]:


x = np.array([0,10])

err = [np.linalg.norm(x - xstar)]
for k in range(25):

    # gradient
    s = -df(x)
    
    # search for the best distance
    def f1d(alpha):
        return f(x + alpha*s)
    alpha_opt = sopt.golden(f1d)
    
    # take a step
    x = x + alpha_opt * s
    
    err.append(np.linalg.norm(x - xstar))


# In[68]:


plt.semilogy(err)


# In[ ]:




