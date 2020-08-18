#!/usr/bin/env python
# coding: utf-8

# # Behavior of Elimination Matrices

# In[3]:


import numpy as np


# In[30]:


n = 4


# ----------------
# Let's create some elimination matrices:

# In[40]:


M1 = np.eye(n)
M1[1,0] = 0.5
M1


# In[41]:


M2 = np.eye(n)
M2[3,0] = 4
M2


# In[42]:


M3 = np.eye(n)
M3[2,1] = 1.3
M3


# -------------------
# Now play around with them:

# In[43]:


M1.dot(M2)


# In[44]:


M2.dot(M1)


# In[45]:


M1.dot(M2).dot(M3)


# BUT:

# In[47]:


M3.dot(M1).dot(M2)

