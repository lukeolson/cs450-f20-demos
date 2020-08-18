#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_cell_magic('timeit', '', 's = 0\nfor i in range(100):\n    s += i**2')


# In[3]:


print(s)


# In[4]:


get_ipython().run_cell_magic('timeit', '', 'a = np.arange(100)\ns = (a**2).sum()')


# In[5]:


print(s)


# In[ ]:




