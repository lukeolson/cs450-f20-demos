#!/usr/bin/env python
# coding: utf-8

# # Bauer-Fike Eigenvalue Sensitivity Bound

# In[2]:


import numpy as np
import numpy.linalg as la


# In the Bauer-Fike eigenvalue sensitivity bound, an important observation is that, given a diagonalized matrix
# $$X^{- 1} A  X = D$$
# that is perturbed by an additive perturbation $E$
# $$X^{- 1} (A + E) X = D + F,$$
# and if we suppose that $\mu$ is an eigenvalue of $A+E$ (and $D+F$), we have
# $$\|(\mu I - D)^{- 1}\|^{- 1} = | \mu - \lambda _k |,$$
# where $\lambda_k$ is the eigenvalue of $A$ (diagonal entry of $D$) closest to $\mu$.
# 
# This notebook illustrates this latter fact. To that end, let the following be $D$:

# In[4]:


D = np.diag(np.arange(6))
D


# In[5]:


mu = 2.1


# In[6]:


mu * np.eye(6) - D


# In[9]:


la.inv(mu * np.eye(6) - D).round(3)


# In[10]:


la.norm(la.inv(mu * np.eye(6) - D), 2)


# The actual norm doesn't matter--the norm of a diagonal matrix has to be the biggest (abs. value) diagonal entry:

# In[11]:


la.norm(la.inv(mu * np.eye(6) - D), np.inf)


# In[12]:


1/ la.norm(la.inv(mu * np.eye(6) - D), 2)


# Note that this matches the distance between $\mu$ and the closest entry of $D$.

# In[ ]:




