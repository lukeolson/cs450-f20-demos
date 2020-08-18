#!/usr/bin/env python
# coding: utf-8

# # Monomial interpolation

# In[1]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


# In[2]:


x = np.linspace(0, 1, 200)


# Now plot the monomial basis on the interval [0,1] up to $x^9$.

# In[3]:


n = 10

for i in range(n):
    pt.plot(x, x**i)
    
pt.vlines(np.linspace(0, 1, n), 0, 1, alpha=0.5, linestyle="--")


# * How do the entries of the Vandermonde matrix relate to this plot?

# ------------------
# * Guess the condition number of the Vandermonde matrix for $n=5,10,20$:

# In[5]:


n = 10

V = np.array([np.linspace(0, 1, n)**i for i in range(n)]).T
la.cond(V)


# In[ ]:




