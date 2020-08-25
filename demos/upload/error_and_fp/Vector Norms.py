#!/usr/bin/env python
# coding: utf-8

# # Vector Norms

# ## Computing norms by hand
# 
# $p$-norms can be computed in two different ways in numpy:

# In[1]:


import numpy as np
import numpy.linalg as la


# In[2]:


x = np.array([1.,2,3])


# First, let's compute the 2-norm by hand:

# In[3]:


np.sum(x**2)**(1/2)


# Next, let's use `numpy` machinery to compute it:

# In[4]:


la.norm(x, 2)


# Both of the values above represent the 2-norm: $\|x\|_2$.

# --------------
# 
# ## About the $\infty$-norm
# 
# Different values of $p$ work similarly:

# In[5]:


np.sum(np.abs(x)**5)**(1/5)


# In[6]:


la.norm(x, 5)


# ---------------------
# 
# The $\infty$ norm represents a special case, because it's actually (in some sense) the *limit* of $p$-norms as $p\to\infty$.
# 
# Recall that: $\|x\|_\infty = \max(|x_1|, |x_2|, |x_3|)$.
# 
# Where does that come from? Let's try with $p=100$:

# In[7]:


x**100


# In[8]:


np.sum(x**100)


# Compare to last value in vector: the addition has essentially taken the maximum:

# In[9]:


np.sum(x**100)**(1/100)


# Numpy can compute that, too:

# In[10]:


la.norm(x, np.inf)


# -------------
# 
# ## Unit Balls
# 
# Once you know the set of vectors for which $\|x\|=1$, you know everything about the norm, because of semilinearity. The graphical version of this is called the 'unit ball'.
# 
# We'll make a bunch of vectors in 2D (for visualization) and then scale them so that $\|x\|=1$.

# In[11]:


alpha = np.linspace(0, 2*np.pi, 2000, endpoint=True)
x = np.cos(alpha)
y = np.sin(alpha)

vecs = np.array([x,y])

p = 5
norms = np.sum(np.abs(vecs)**p, axis=0)**(1/p)
norm_vecs = vecs/norms

import matplotlib.pyplot as pt
get_ipython().run_line_magic('matplotlib', 'inline')
pt.grid()
pt.gca().set_aspect("equal")
pt.plot(norm_vecs[0], norm_vecs[1])
pt.xlim([-1.5, 1.5])
pt.ylim([-1.5, 1.5])


# In[ ]:





# In[ ]:





# In[ ]:




