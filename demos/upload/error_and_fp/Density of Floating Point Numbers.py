#!/usr/bin/env python
# coding: utf-8

# # Density of Floating Point Numbers
# 
# This notebook enumerates all possible floating point nubmers in a floating point system and shows them in a plot to illustrate their density.

# In[2]:


import matplotlib.pyplot as pt
import numpy as np


# In[13]:


significand_bits = 4
exponent_min = -3
exponent_max = 4

fp_numbers = []
for exp in range(exponent_min, exponent_max+1):
    for sbits in range(0, 2**significand_bits):
        significand = 1 + sbits/2**significand_bits 
        fp_numbers.append(significand * 2**exp)
        
fp_numbers = np.array(fp_numbers)
print(fp_numbers)

pt.plot(fp_numbers, np.ones_like(fp_numbers), "+")
#pt.semilogx(fp_numbers, np.ones_like(fp_numbers), "+")
        


# In[ ]:




