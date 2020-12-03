#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')


# In[2]:


A = np.array([[0.8, 0.6, 0.8],
              [0.2, 0.3, 0.0],
              [0.0, 0.1, 0.2]])


# In[3]:


v = np.random.rand(3)
v = v / np.linalg.norm(v,1)
print(v)
print(v.sum())


# In[13]:


x = v.copy()

n = 10
eigs = np.zeros(n)
eigv = np.zeros((3,n))

for i in range(n):
    y = A @ x
    normy = np.linalg.norm(y, np.inf)
    x = y / normy
    #y = x
    eigs[i] = normy
    eigv[:,i] = x


# In[10]:


plt.plot(eigv.T);
plt.legend(['surf', 'study', 'eat'])


# In[11]:


x


# In[12]:


x / np.linalg.norm(x,1)


# #  Page Rank

# A = array([ [0,     0,     0,     1, 0, 1],
#             [1/2.0, 0,     0,     0, 0, 0],
#             [0,     1/2.0, 0,     0, 0, 0],
#             [0,     1/2.0, 1/3.0, 0, 0, 0],
#             [0,     0,     1/3.0, 0, 0, 0],
#             [1/2.0, 0,     1/3.0, 0, 1, 0 ] ])

# In[14]:


A = np.array([ [0,     0,     0,     1, 0, 1],
               [1/2.0, 0,     0,     0, 0, 0],
               [0,     1/2.0, 0,     0, 0, 0],
               [0,     1/2.0, 1/3.0, 0, 0, 0],
               [0,     0,     1/3.0, 0, 0, 0],
               [1/2.0, 0,     1/3.0, 0, 1, 0]])


# In[15]:


u = np.ones(6)
M = 0.85 * A + 0.15 * (1/6) * np.outer(u,u)


# In[16]:


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


# In[17]:


x / np.linalg.norm(x,1)


# In[19]:


plt.plot(eigv.T)
plt.legend([f'{i}' for i in range(n)])


# In[ ]:


eigs


# In[ ]:


w, v = np.linalg.eig(M)


# In[ ]:


w


# In[ ]:


v = v[:,0]


# In[ ]:


v = -v / np.linalg.norm(v,1)


# In[ ]:


v


# In[ ]:




