#!/usr/bin/env python
# coding: utf-8

# # Householder Similarity Transforms

# In[1]:


import numpy as np
import numpy.linalg as la

np.set_printoptions(precision=2, linewidth=150)


# In[2]:


n = 8

e1 = np.zeros(n); e1[0] = 1
e2 = np.zeros(n); e2[1] = 1

A = np.random.randn(n, n)
A


# Now try to zero the first column with a similarity transform.
# 
# ## Starting with the first row
# 
# Lets first try to proceed as in Householder QR, creating a transformation
# $$H=I-2\frac{vv^T}{v^Tv}$$
# where $v = a_1 - ||a_1||_2e_1$ with $a_1$ being the first column of $A$.

# In[3]:


a = A[:, 0].copy()
v = a-la.norm(a)*e1

H1 = np.eye(n) - 2*np.outer(v, v)/(v@v)


# We can apply the transformation from the left as in QR to reduce the first column to a multiple of the first elementary vector.

# In[4]:


(H1@A).round(4)


# However, to ensure we do not perturb the eigenvalues of $A$, we must also apply the matrix from the right, resulting in a similarity transformation.

# In[5]:


(H1@A@H1.T).round(4)


# In[7]:


H1


# Note that applying the Householder transformation from the right filled in the elements annihilated by applying it from the left.
# 
# ## Starting in the second row
# 
# To avoid this, we define the Householder transformation to annihilate elements below the first subdiagonal. That way, the first transformation does not affect the first row when applied from the left, and consequently does not affect the first column when applied for the right, preserving the zeros we've annihilated.

# In[8]:


a = A[:, 0].copy()
a[0] = 0
v = a-la.norm(a)*e2

H2 = np.eye(n) - 2*np.outer(v, v)/(v@v)


# In[9]:


H2


# In[10]:


(H2 @ A).round(4)


# In[11]:


(H2 @ A @ H2.T).round(4)


# To generalize this process, we continue to eliminate everything below the subdiagonal in the next column and applying the two-sided transformations, finally resulting in an upper-Hessenberg matrix.
# 
# -----
# 
# Why does post-multiplying with `H2.T` not destroy the zeros?

# In[ ]:


H2.T.round(4)


# In[ ]:


v


# In[ ]:


H2


# In[ ]:




