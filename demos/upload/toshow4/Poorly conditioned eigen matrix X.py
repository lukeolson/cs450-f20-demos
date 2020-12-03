#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[15]:


n = 15
x = np.linspace(0,1,n)
X = np.zeros((n,n))
for i in range(n):
    X[:,i] = x**i


# In[16]:


print(X)


# In[17]:


for i in range(n):
    plt.plot(x, X[:,i])


# In[18]:


A = np.linalg.inv(X) @ np.diag(np.arange(1,n+1)) @ X


# In[19]:


np.linalg.cond(A)


# In[13]:


w, _ = np.linalg.eig(A)
print(w)


# In[ ]:





# In[ ]:





# In[ ]:




