#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-talk')


# In[2]:


def power_method(A, x, maxiter):
    eigs = np.zeros(maxiter)
    eigv = np.zeros((len(x), maxiter))

    for i in range(maxiter):
        y = A @ x
        ynorm = np.linalg.norm(y,np.inf)
        x = y / ynorm
        
        eigs[i] = ynorm
        eigv[:,i] = x
        
    return x, eigs, eigv


# # Test 1

# In[ ]:


n = 12
x = np.linspace(0,1,n)
X = np.zeros((n,n))
for i in range(n):
    X[:,i] = x**i
A = np.linalg.inv(X) @ np.diag(np.arange(1,n+1)) @ X

x, eigs, eigv = power_method(A, np.random.rand(A.shape[0]), 1000)


# In[ ]:


eigs


# In[ ]:


[(11/12)**k for k in range(100)]


# # Test 2

# In[ ]:


n = 10
X = np.random.rand(n,n)
A = X @ np.diag(np.arange(1,n+1)) @ np.linalg.inv(X)

x0 = np.random.rand(10)
#x0 = X[:,:-1].sum(axis=1)

x, eigs, eigv = power_method(A, x0, 50)


# In[ ]:


eigs


# # Test 3

# In[ ]:


n = 10
X = np.random.rand(n,n)
diag = np.arange(1,n+1)
diag[-2] = -10
D = np.diag(diag)
A = X @ D @ np.linalg.inv(X)

x0 = np.random.rand(10)

x, eigs, eigv = power_method(A, x0, 10)


# In[ ]:


eigs


# In[ ]:


plt.plot(eigv.T)


# # Test 4

# In[ ]:


n = 5
A = np.random.rand(n,n)

x0 = np.random.rand(5)

x, eigs, eigv = power_method(A, x0, 10)


# In[ ]:


eigs


# In[ ]:


x


# In[ ]:


np.linalg.eig(A)


# # Test 5

# In[ ]:


n = 10
X = np.random.rand(n,n)
diag = np.arange(1,n+1)
diag[-2] = 8
D = np.diag(diag)
A = X @ D @ np.linalg.inv(X)

x0 = np.random.rand(10)

x, eigs, eigv = power_method(A, x0, 100)


# In[ ]:


plt.semilogy(np.abs(eigs-10))


# In[ ]:


n = 10
X = np.random.rand(n,n)
diag = np.arange(1,n+1)
diag[-2] = 9
D = np.diag(diag)
A = X @ D @ np.linalg.inv(X)

x0 = np.random.rand(10)

x, eigs, eigv = power_method(A, x0, 100)


# In[ ]:


plt.semilogy(np.abs(eigs-10))


# # Test 6
# 
# Shift!

# In[ ]:


n = 10
X = np.random.rand(n,n)
diag = np.arange(1,n+1)
diag[-2] = 9
D = np.diag(diag)
A = X @ D @ np.linalg.inv(X)

x0 = np.random.rand(10)

x, eigs, eigv = power_method(A - 5*np.eye(n), x0, 100)


# In[ ]:


plt.semilogy(np.abs(eigs-(10-5)))


# In[ ]:




