#!/usr/bin/env python
# coding: utf-8

# # Gaussian elimination

# In[3]:


import numpy as np


# In[18]:



np.random.seed(5)
n = 4
A = np.round(np.random.randn(n, n) * 5)
A


# Now compute `A1` to eliminate `A[1,0]`:

# In[23]:


A1 = A.copy()
A1[1] -= 1/2*A1[0]
A1


# And `A2` with `A[2,0] == 0`:

# In[25]:


A2 = A1.copy()
A2[2] -= 1/2*A[0]
A2


# In[ ]:




