#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# In[8]:


get_ipython().run_cell_magic('timeit', '', 's = 0\nfor i in range(1000000):\n    s += i**2')


# In[5]:


print(s)


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'a = np.arange(1000000)\ns = (a**2).sum()')


# In[ ]:


print(s)


# In[ ]:




