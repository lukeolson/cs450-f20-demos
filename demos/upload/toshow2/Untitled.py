#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pyamg
import numpy as np


# In[2]:


get_ipython().system('pip install pyamg')


# In[4]:


A = pyamg.gallery.poisson((5,5), format='csr')


# In[5]:


A


# In[21]:


I = np.where(np.diff(A.indptr)==4)[0]


# In[22]:


I


# In[32]:


B = A[I,:][:,I]


# In[24]:


B


# In[25]:


B.shape


# In[26]:


A.spae


# In[27]:


A.shape


# In[33]:


B


# In[31]:


B.toarray()


# In[34]:


np.diff(B.indptr)


# In[ ]:




