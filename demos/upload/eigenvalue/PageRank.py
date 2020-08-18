#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')


# In[12]:


A = np.array([[0.8, 0.6, 0.8],
              [0.2, 0.3, 0.0],
              [0.0, 0.1, 0.2]])


# In[13]:


v = np.random.rand(3)
v = v / np.linalg.norm(v,1)
print(v)
print(v.sum())


# In[25]:


x = v.copy()

n = 10
eigs = np.zeros(n)
eigv = np.zeros((3,n))

for i in range(n):
    y = A @ x
    normy = np.linalg.norm(y, np.inf)
    x = y / normy
    eigs[i] = normy
    eigv[:,i] = x


# In[26]:


plt.plot(eigv.T);


# In[27]:


x


# In[28]:


x / np.linalg.norm(x,1)


# #  Page Rank

# A = array([ [0,     0,     0,     1, 0, 1],
#             [1/2.0, 0,     0,     0, 0, 0],
#             [0,     1/2.0, 0,     0, 0, 0],
#             [0,     1/2.0, 1/3.0, 0, 0, 0],
#             [0,     0,     1/3.0, 0, 0, 0],
#             [1/2.0, 0,     1/3.0, 0, 1, 0 ] ])

# In[30]:


A = np.array([ [0,     0,     0,     1, 0, 1],
               [1/2.0, 0,     0,     0, 0, 0],
               [0,     1/2.0, 0,     0, 0, 0],
               [0,     1/2.0, 1/3.0, 0, 0, 0],
               [0,     0,     1/3.0, 0, 0, 0],
               [1/2.0, 0,     1/3.0, 0, 1, 0]])


# In[103]:


u = np.ones(6)
M = 0.85 * A + 0.15 * (1/6) * np.outer(u,u)


# In[126]:


x = np.random.rand(6)
x = x / np.linalg.norm(x,1)
#x = np.ones(6)/6

n = 15
eigs = np.zeros(n)
eigv = np.zeros((6,n))

for i in range(n):
    y = M @ x
    normy = np.linalg.norm(y, np.inf)
    x = y / normy
    eigs[i] = normy
    eigv[:,i] = x / np.linalg.norm(x,1)


# In[127]:


x / np.linalg.norm(x,1)


# In[128]:


plt.plot(eigv.T)


# In[118]:


eigs


# In[107]:


w, v = np.linalg.eig(M)


# In[108]:


w


# In[109]:


v = v[:,0]


# In[110]:


v = -v / np.linalg.norm(v,1)


# In[111]:


v


# In[ ]:




