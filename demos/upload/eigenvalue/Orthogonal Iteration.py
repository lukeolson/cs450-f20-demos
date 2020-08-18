#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context('talk')


# In[10]:


np.random.seed(2222)
T, _ = np.linalg.qr(np.random.rand(5,5))
A = T.T @ np.diag([5,4,3,2,1]) @ T

w, v = np.linalg.eig(A)
np.set_printoptions(precision=2)
print(w)
for ww in w:
    print(ww, np.abs(ww))
print(v)


# In[11]:


X = np.random.rand(5,3)
for i in range(50):
    Q, R = np.linalg.qr(X, mode='reduced')
    X = A @ Q
    
    print(np.diag(X.conj().T @ (A @ X)) / np.diag(X.conj().T @ X))


# # QR

# In[12]:


X = A.copy()
Qall = np.eye(5)
for i in range(10):
    Q, R = np.linalg.qr(X)
    X = R @ Q
    
    np.set_printoptions(precision=2)
    print(np.diag(X))
    
    #Qall = Qall @ Q
    #print(Qall)


# In[13]:


from matplotlib.colors import LogNorm


# In[78]:


np.random.seed(12112019)
niter = 400
nprint = 5
niterprint = np.floor(niter / nprint)

A = np.random.rand(10,10)
A = A + 1j * np.random.rand(10,10)
A = A + A.T
X = A.copy()

ct = 0
f, ax = plt.subplots(2,nprint,sharey=True,figsize=(12,6))
for i in range(niter):
    Q, R = np.linalg.qr(X)
    X = R @ Q
    
    if i % niterprint == 0:
        
        I, J = np.where(np.abs(X) < 1e-13)
        Xtmp = X.copy()
        Xtmp[I,J] = 0.0
        
        im = ax[0,ct].imshow(np.abs(Xtmp.real), cmap=plt.cm.winter, norm=LogNorm())
        ax[0,ct].axis('off')
        
        if np.abs(Xtmp.imag).max() > 1e-13:
            im = ax[1,ct].imshow(np.abs(Xtmp.imag), cmap=plt.cm.winter, norm=LogNorm())
        ax[1,ct].axis('off')
        ct += 1
        
f.colorbar(im, ax=ax.ravel().tolist(), shrink=0.95)


# In[79]:


Xtmp


# In[ ]:




