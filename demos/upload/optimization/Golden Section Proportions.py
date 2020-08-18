#!/usr/bin/env python
# coding: utf-8

# # Proportions of the Golden Section

# In[20]:


import matplotlib.pyplot as pt
from math import sqrt


# In[33]:


a = 0
b = 1

m1 = a + (1-(sqrt(5)-1)/2) * (b-a)
m2 = a + (sqrt(5)-1)/2 * (b-a)

pt.xlim([a-0.5, b+0.5])
pt.grid()
pt.plot([a,b], [0,0], "ob")
pt.plot([m1, m2], [0,0], "or")

a = m1
m1 = a + (1-(sqrt(5)-1)/2) * (b-a)
m2 = a + (sqrt(5)-1)/2 * (b-a)
pt.plot([m1, m2], [0,0], "ko", marker='o', alpha=.25, ms=15)


# In[ ]:




