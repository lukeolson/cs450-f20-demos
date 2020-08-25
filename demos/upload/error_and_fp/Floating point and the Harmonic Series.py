#!/usr/bin/env python
# coding: utf-8

# # Floating Point and the Harmonic Series
# 
# You may know from math that
# $$
# \sum_{n=1}^\infty \frac 1n=\infty.
# $$
# Let's see what we get using floating point:

# In[1]:


import numpy as np


# In[6]:


n = int(0)

float_type = np.float32

my_sum = float_type(0)

while True:
    n += 1
    last_sum = my_sum
    my_sum += float_type(1 / n)
    
    if n % 200000 == 0:
        print("1/n = %g, sum0 = %g"%(1.0/n, my_sum))
        
    if my_sum == last_sum:
        print("1/n = %g, sum0 = %g"%(1.0/n, my_sum))
        break


# In[2]:




