#!/usr/bin/env python
# coding: utf-8

# # Rates of Convergence

# In[1]:


import numpy as np


# In[2]:


C = 1/2
e0 = 0.1*np.random.rand()

rate = 1


# In[3]:


e = e0
for i in range(20):
    print(e)
    e = C*e**rate


# * What do you observe about the number of iterations it takes to decrease the error by a factor of 10 for `rate=1,2,3`?
# * Is there a point to faster than cubic convergence?

# ------------------
# Now let's see if we can figure out the convergence order from the data.
# 
# Here's a function that spits out some fake errors of a process that converges to $q$th order:

# In[4]:


def make_rate_q_errors(q, e0, n=10, C=0.7):
    errors = []
    e = e0
    for i in range(n):
        errors.append(e)
        e = C*e**q
        
    return errors


# In[5]:


errors = make_rate_q_errors(1, e0)


# In[6]:


for e in errors:
    print(e)


# In[7]:


for i in range(len(errors)-1):
    print(errors[i+1]/errors[i])

