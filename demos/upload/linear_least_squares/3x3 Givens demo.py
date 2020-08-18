#!/usr/bin/env python
# coding: utf-8

# # 3x3 Givens Rotation

# In[4]:


import numpy as np


# In[5]:


a = np.random.randn(3)
a


# Let's zero out $a_2$:

# In[7]:


G = np.zeros((3, 3))

c = a[0]/np.sqrt(a[0]**2 + a[1]**2)
s = a[1]/np.sqrt(a[0]**2 + a[1]**2)

G[:2,:2] = np.array([
        [c, s],
        [-s, c]
        ])
G


# Anything wrong with $G$?

# <!--
# G zeroes out the last component.
# -->
# (Edit this cell for solution.)

# How would we fix that issue?

# In[11]:


G[2,2] = 1


# In[10]:


G @ a


# Observations?
