#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sp
sp.init_printing()


# In[2]:


eps = sp.Symbol("epsilon")
lam = sp.Symbol("lambda")


# In[3]:


m = sp.Matrix([[1, eps], [eps, 1]])
m


# In[4]:


m.charpoly(lam)


# Observe the occurrence of $(1-\epsilon^2)$ above.
