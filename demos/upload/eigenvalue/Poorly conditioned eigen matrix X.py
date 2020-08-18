#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[24]:


n = 12
x = np.linspace(0,1,n)
X = np.zeros((n,n))
for i in range(n):
    X[:,i] = x**i


# In[25]:


A = np.linalg.inv(X) @ np.diag(np.arange(1,n+1)) @ X


# In[26]:


np.linalg.cond(A)


# In[29]:


w, _ = np.linalg.eig(A)
print(w)


# In[ ]:





# In[ ]:




