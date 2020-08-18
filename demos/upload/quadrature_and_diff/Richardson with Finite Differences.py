#!/usr/bin/env python
# coding: utf-8

# # Using Richardson Extrapolation with Finite Differences

# In[1]:


from math import sin, cos


# Here are a function and its derivative. We also choose a "center" about which we carry out our experiments:

# In[2]:


f = sin
df = cos

x = 2.3


# We then compare the accuracy of:
# 
# * First-order (right) differences
# * First-order (right) differences with half the step size
# * An estimate based on these two using Richardson extrapolation
# 
# against `true`, the actual derivative

# In[3]:


for k in range(3, 10):
    h = 2**(-k)

    h1 = 2*h
    fd1 = (f(x+h1) - f(x))/(h1)
    
    h2 = h
    fd2 = (f(x+h2) - f(x))/h2
    
    p = 1
    alpha = - h2**p / (h1**p - h2**p)
    beta = 1 - alpha
    richardson = alpha*fd1 + beta*fd2
    
    true = df(x)
    
    print("Err FD1: %g\tErr FD: %g\tErr Rich: %g" % (
            abs(true-fd1),
            abs(true-fd2),    
            abs(true-richardson)))


# In[4]:


3.39995e-06 / 8.48602e-07


# In[ ]:




