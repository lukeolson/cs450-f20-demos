#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


errs = []
conds = []

xstar = np.ones(2)
for eps in np.logspace(-1,-14):
    
    A = np.array([[1, 0], [0, eps]])
    V,_ = la.qr(np.random.randn(2,2))
    A = V.T @ A @ V
    b = A.dot(xstar)
    x = np.linalg.solve(A, b)
    diff = np.max(np.abs(x-xstar))
    cond = np.linalg.cond(A)
    errs.append(diff)
    conds.append(cond)
    print("{:20.2e} {:20.2e}".format(cond, diff))


# In[41]:


plt.loglog(conds, errs, 'o')


# In[17]:


xstar


# In[18]:


b


# In[ ]:




