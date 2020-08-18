#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.set_printoptions(precision=2)


# ### Case 1: General, Non-singular matrix

# In[ ]:


A = np.random.rand(4,4)
print(A)


# In[ ]:


P, L, U = sla.lu(A)
print(P)
print(L)
print(U)
print(A - P@L@U)


# In[ ]:


A[piv]


# ### Case 2: diagonally dominant matrix

# In[ ]:


A = np.random.rand(4,4) + 10 * np.eye(4)
print(A)


# In[ ]:


P, L, U = sla.lu(A)
print(P)
print(L)
print(U)


# ### Case 3: symmetric, positive definite matrix

# In[ ]:


A = np.random.rand(4,4)
A = A.T @ A
print(A)


# In[ ]:


P, L, U = sla.lu(A)
print(P)
print(L)
print(U)


# ### Case 4: symmetric, positive definite matrix, redo

# In[ ]:


A = np.random.rand(4,4)
A = A.T @ A
print(A)


# In[ ]:


L = sla.cholesky(A)
print(L)


# In[ ]:


print(L.T @ L)


# In[ ]:


A = np.random.rand(3000,3000)
A = A.T @ A


# In[ ]:


get_ipython().run_line_magic('timeit', 'PLU = sla.lu(A)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'L = sla.cholesky(A)')


# ### Case 5: singular matrix

# In[ ]:


D = np.diag([1,2,1e-13,0,0])+np.tril(np.random.rand(5,5),k=-1)
A = D
#V, _ = np.linalg.qr(np.random.rand(5,5))
#A = V.T @ D @ V
print(A)
print("The eigenvalues: ", np.linalg.eig(A)[0])


# In[ ]:


P, L, U = sla.lu(np.array([[0,1],[1,1]]))
print(P)
print(L)
print(U)


# In[ ]:


P, L, U = sla.lu(A)
print(P)
print(L)
print(U)


# ### Case 6: long matrix

# In[ ]:


A = np.random.rand(4, 8)
print(A)


# In[ ]:


P, L, U = sla.lu(A)


# In[ ]:


print(L)
print(U)


# ### Case 7: tall matrix

# In[ ]:


A = np.random.rand(8, 4)
print(A)


# In[ ]:


P, L, U = sla.lu(A)
print(L)
print('---')
print(U)


# In[ ]:


sla.lu_factor(np.zeros((3,3)))


# In[ ]:




