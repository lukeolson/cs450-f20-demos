#!/usr/bin/env python
# coding: utf-8

# # Newton's method in 1D

# In[1]:


import numpy as np
import matplotlib.pyplot as pt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_context('talk')


# Here's a function:

# In[2]:


a = 17.09
b = 9.79
c = 0.6317
d = 0.9324
e = 0.4565

def f(x):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def df(x):
    return 4*a*x**3 + 3*b*x**2 + 2*c*x + d

def d2f(x):
    return 3*4*a*x**2 + 2*3*b*x + 2*c


# Let's plot the thing:

# In[3]:


xmesh = np.linspace(-1, 0.5
                    , 100)
pt.ylim([-1, 3])
pt.plot(xmesh, f(xmesh))


# Let's fix an initial guess:

# In[4]:


x = 0.3


# In[9]:


dfx = df(x)
d2fx = d2f(x)

# carry out the Newton step
xnew = x - dfx / d2fx

# plot approximate function
pt.plot(xmesh, f(xmesh))
pt.plot(xmesh, f(x) + dfx*(xmesh-x) + d2fx*(xmesh-x)**2/2)
pt.plot(x, f(x), "o", color="red")
pt.plot(xnew, f(xnew), "o", color="green")
pt.ylim([-1, 3])

# update
x = xnew
print(x)


# * What convergence order does this method achieve?

# In[ ]:


# Quadratic, because it's just like doing 'equation-solving Newton' on f'.

