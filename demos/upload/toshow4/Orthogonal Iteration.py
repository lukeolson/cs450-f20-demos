#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-talk')


# In[3]:


np.random.seed(2222)
T, _ = np.linalg.qr(np.random.rand(5,5))
A = T.T @ np.diag([5,4,3,2,1]) @ T
print(A.T - A)
w, v = np.linalg.eig(A)
np.set_printoptions(precision=2)
print(w)
for ww in w:
    print(ww, np.abs(ww))
print(v)


# In[8]:


#A = np.random.rand(5,5)
X = np.random.rand(5,3)
for i in range(50):
    Q, R = np.linalg.qr(X, mode='reduced')
    X = A @ Q
    print(Q.conj().T @ A @ Q)
    print('---------')
    #print(np.diag(Q.conj().T @ (A @ Q)))


# In[ ]:


Q.T @ A @ Q


# # QR

# In[9]:


X = A.copy()
Qall = np.eye(5)
for i in range(10):
    Q, R = np.linalg.qr(X)
    X = R @ Q # X = Ak = Q.T A Q -> real diagonal (for a real, symm A)
    
    np.set_printoptions(precision=2)
    print(np.diag(X))
    
    #Qall = Qall @ Q
    #print(Qall)


# In[10]:


from matplotlib.colors import LogNorm


# In[17]:


np.random.seed(12112019)
niter = 800
nprint = 5
niterprint = np.floor(niter / nprint)

A = np.random.rand(10,10)
#A = A + 1j * np.random.rand(10,10)
#A = A + A.T
X = A.copy()

ct = 0
f, ax = plt.subplots(2,nprint,sharey=True,figsize=(12,6))
for i in range(niter):
    Q, R = np.linalg.qr(X)
    X = R @ Q # X = Ak = Q.T A Q
    
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


# In[19]:


plt.imshow(np.abs(Xtmp.real), cmap=plt.cm.winter, norm=LogNorm())


# In[20]:


np.linalg.eig(A)


# In[23]:


np.linalg.eig(X[1:3,1:3])


# In[21]:


Xtmp[0,0]


# In[22]:


Xtmp[9,9]


# In[ ]:




