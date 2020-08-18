#!/usr/bin/env python
# coding: utf-8

# # Floating Point Arithmetic and the Series for the Exponential Function

# In[4]:


import numpy as np
import matplotlib.pyplot as pt


# What this demo does is sum the series
# $$
#   \exp(x) \approx \sum_{i=0}^n \frac{x^i}{i!},
# $$
# for varying $n$, and varying $x$. It then prints the partial sum, the true value, and the final term of the series.

# In[5]:


a = 0.0
x = 1e0 # flip sign
true_f = np.exp(x)
e = []

for i in range(0, 10): # crank up
    d = np.prod(
            np.arange(1, i+1).astype(np.float))
    
    # series for exp
    a += x**i / d

    print(a, np.exp(x), x**i / d)
    
    e.append(abs(true_f-a)/true_f)


# In[6]:


pt.semilogy(e)


# In[3]:




