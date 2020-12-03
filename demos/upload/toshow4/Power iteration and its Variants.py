#!/usr/bin/env python
# coding: utf-8

# # Power Iteration and its Variants

# In[33]:


import numpy as np
import numpy.linalg as la
np.set_printoptions(precision=3)


# Let's  prepare a matrix with some random or deliberately chosen eigenvalues:

# In[36]:


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

# In[37]:


x0 = np.random.randn(n)
x0


# ### Power iteration

# In[42]:


x = x0


# Now implement plain power iteration.

# In[52]:


for i in range(20):
    xold = x.copy()
    x = A @ x
    #print(x / xold)
    print('|eig| ~', x[0] / xold[0])


# * What's the problem with this method?
# * Does anything useful come of this?
# * How do we fix it?

# ### Normalized power iteration

# Back to the beginning: Reset to the initial vector.

# In[53]:


x = x0/la.norm(x0)


# Implement normalized power iteration.

# In[54]:


for i in range(10):
    x = A @ x
    nrm = la.norm(x)
    x = x/nrm
    print('|eig| ~', nrm)

print(x)


# In[55]:


A @ x - - nrm * x


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

# In[56]:


x = x0/la.norm(x0)


# In[57]:


for i in range(30):
    x = la.solve(A, x)
    nrm = la.norm(x)
    x = x/nrm
    print('|eig| ~', 1/nrm)
print(nrm)


# * What's the computational cost per iteration?
# * Can we make this method search for a specific eigenvalue?

# ------
# ### Inverse Shift iteration
# 
# What if we want the eigenvalue closest to a give value $\sigma$?
# 
# Once again, reset to the beginning.

# In[64]:


x = x0/la.norm(x0)


# In[69]:


sigma = -1
A_sigma = A-sigma*np.eye(A.shape[0])
for i in range(30):
    x = la.solve(A_sigma, x)
    nrm = la.norm(x)
    x = x/nrm
    print('|eig| ~', 1/nrm)
print(nrm)
print(eigvals - sigma)


# --------------
# ### Rayleigh quotient iteration
# 
# Can we feed an estimate of the current approximate eigenvalue back into the calculation? (Hint: Rayleigh quotient)
# 
# Reset once more.

# In[75]:


x = x0/la.norm(x0)


# Run this cell in-place (Ctrl-Enter) many times.

# In[77]:


for i in range(10):
    sigma = np.dot(x, np.dot(A, x))/np.dot(x, x)
    x = la.solve(A-sigma*np.eye(n), x)
    x = x/la.norm(x)
    print('|eig| ~', sigma)
print(eigvals)


# * What's a reasonable stopping criterion?
# * Computational downside of this iteration?
