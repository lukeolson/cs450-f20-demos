#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


A = np.random.rand(5, 5)
x = np.array([1.,2,3,4,5])
b = A @ x


# In[3]:


P, L, U = sla.lu(A)


# In[5]:


A - P @ L @ U


# In[7]:


P.T @ P


# In[10]:


y = sla.solve_triangular(L, P.T @ b, lower=True, unit_diagonal=True)
x = sla.solve_triangular(U, y)
x


# In[ ]:


P


# In[ ]:


P @ b


# In[4]:


get_ipython().set_next_input('y = sla.lu');get_ipython().run_line_magic('pinfo', 'sla.lu')


# In[ ]:


y = sla.lu


# In[ ]:




