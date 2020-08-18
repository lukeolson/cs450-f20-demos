#!/usr/bin/env python
# coding: utf-8

# # Fixed Point Iteration

# In[7]:


import numpy as np
import matplotlib.pyplot as pt


# **Task:** Find a root of the function below by fixed point iteration.

# In[8]:


x = np.linspace(0, 4.5, 200)

def f(x):
    return x**2 - x - 2

pt.plot(x, f(x))
pt.grid()


# Actual roots: $2$ and $-1$. Here: focusing on $x=2$.

# We can choose a wide variety of functions that have a fixed point at the root $x=2$:
# 
# (These are chosen knowing the root. But here we are only out to study the *behavior* of fixed point iteration, not the finding of fixed point functions--so that is OK.)

# In[9]:


def fp1(x): return x**2-2
def fp2(x): return np.sqrt(x+2)
def fp3(x): return 1+2/x
def fp4(x): return (x**2+2)/(2*x-1)

fixed_point_functions = [fp1, fp2, fp3, fp4]


# In[11]:


for fp in fixed_point_functions:
    pt.plot(x, fp(x), label=fp.__name__)
pt.ylim([0, 3])
pt.legend(loc="best")


# Common feature?

# In[13]:


for fp in fixed_point_functions:
    print(fp(2))
    
# All functions have 2 as a fixed point.


# In[21]:


z = 2.1; fp = fp1
#z = 1; fp = fp2
#z = 1; fp = fp3
#z = 1; fp = fp4

n_iterations = 4

pt.figure(figsize=(8,8))
pt.plot(x, fp(x), label=fp.__name__)
pt.plot(x, x, "--", label="$y=x$")
pt.gca().set_aspect("equal")
pt.ylim([-0.5, 4])
pt.legend(loc="best")

for i in range(n_iterations):
    z_new = fp(z)
    
    pt.arrow(z, z, 0, z_new-z)
    pt.arrow(z, z_new, z_new-z, 0)
    
    z = z_new
    print(z)


# In[ ]:




