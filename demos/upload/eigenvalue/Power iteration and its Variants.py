#!/usr/bin/env python
# coding: utf-8

# # Power Iteration and its Variants

# In[1]:


import numpy as np
import numpy.linalg as la
np.set_printoptions(precision=3)


# Let's  prepare a matrix with some random or deliberately chosen eigenvalues:

# In[2]:


n = 6

if 1:
    np.random.seed(70)
    eigvecs = np.random.randn(n, n)
    eigvals = np.sort(np.random.randn(n))
    # Uncomment for near-duplicate largest-magnitude eigenvalue
    # eigvals[1] = eigvals[0] + 1e-3

    A = eigvecs.dot(np.diag(eigvals)).dot(la.inv(eigvecs))
    print(eigvals)
    
else:
    # Complex eigenvalues
    np.random.seed(40)
    A = np.random.randn(n, n)
    print(la.eig(A)[0])


# Let's also pick an initial vector:

# In[3]:


x0 = np.random.randn(n)
x0


# ### Power iteration

# In[4]:


x = x0


# Now implement plain power iteration.

# In[5]:


for i in range(20):
    x = A @ x
    print(x)


# * What's the problem with this method?
# * Does anything useful come of this?
# * How do we fix it?

# ### Normalized power iteration

# Back to the beginning: Reset to the initial vector.

# In[6]:


x = x0/la.norm(x0)


# Implement normalized power iteration.

# In[7]:


for i in range(10):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print(x)

print(nrm)


# * What do you observe about the norm?
# * What about the sign?
# * What is the vector $x$ now?
# 
# Extensions:
# 
# * Now try the matrix variants above.
# * Suggest a better way of estimating the eigenvalue. [Hint](https://en.wikipedia.org/wiki/Rayleigh_quotient)

# ------
# ### Inverse iteration
# 
# What if we want the smallest eigenvalue (by magnitude)?
# 
# Once again, reset to the beginning.

# In[8]:


x = x0/la.norm(x0)


# In[9]:


for i in range(30):
    x = la.solve(A, x)
    nrm = la.norm(x)
    x = x/nrm
    print(x)


# * What's the computational cost per iteration?
# * Can we make this method search for a specific eigenvalue?

# ------
# ### Inverse Shift iteration
# 
# What if we want the eigenvalue closest to a give value $\sigma$?
# 
# Once again, reset to the beginning.

# In[10]:


x = x0/la.norm(x0)


# In[11]:


sigma = 1
A_sigma = A-sigma*np.eye(A.shape[0])
for i in range(30):
    x = la.solve(A_sigma, x)
    nrm = la.norm(x)
    x = x/nrm
    print(x)


# --------------
# ### Rayleigh quotient iteration
# 
# Can we feed an estimate of the current approximate eigenvalue back into the calculation? (Hint: Rayleigh quotient)
# 
# Reset once more.

# In[12]:


x = x0/la.norm(x0)


# Run this cell in-place (Ctrl-Enter) many times.

# In[13]:


for i in range(10):
    sigma = np.dot(x, np.dot(A, x))/np.dot(x, x)
    x = la.solve(A-sigma*np.eye(n), x)
    x = x/la.norm(x)
    print(x)


# * What's a reasonable stopping criterion?
# * Computational downside of this iteration?
