#!/usr/bin/env python
# coding: utf-8

# # Finite Differences for Boundary Value Problems

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.sparse as sps


# We'll solve
# 
# $u''+1000(1+x^2)u=0$ on $(-1,1)$
# 
# with $u(-1)=3$ and $u(1)=-3$.

# In[24]:


n = 16
#n = 200

mesh = np.linspace(-1, 1, n)
h = mesh[1] - mesh[0]


# Use `sps.diags(values, offsets=..., shape=(n, n))` to make a centered difference matrix.

# In[25]:


A = sps.diags(
    [1,-2,1],
    offsets=[-1,0,1], 
    shape=(n, n))

if n < 10:
    print(A.toarray())


# In[26]:


diag = np.hstack(([1],-2/h**2 + 1000*(1 + mesh[1:-1]**2), [1]))
z = np.hstack(([0], np.ones(n-3)/h**2, [0]))

A = sps.diags(
    [z,diag,z],
    offsets=[-1,0,1],
    shape=(n, n)
)

if n < 10:
    np.set_printoptions(precision=1)
    print(A.toarray())


# In[27]:


plt.spy(A)


# Next, assemble the right-hand side as `rhs`:
# 
# Pay special attention to the boundary conditions. What entries of `rhs` do they correspond to?

# In[21]:


rhs = np.zeros(n)
rhs[0] = 3
rhs[1] = -3/h**2
rhs[-2] = 3/h**2
rhs[-1] = -3


# To wrap up, solve and plot:

# In[22]:


import scipy.sparse.linalg as sla

sol = sla.spsolve(A, rhs)


# In[23]:


plt.plot(mesh, sol)


# In[29]:


A
B = A.tocoo()


# In[30]:


B


# In[31]:


B.nnz


# In[32]:


B.shape


# In[33]:


16*16


# In[34]:


B.row


# In[35]:


B.col


# In[36]:


B.data


# In[ ]:


A.to

