#!/usr/bin/env python
# coding: utf-8

# # Stability Experiments for Backward Euler

# In[1]:


import numpy as np
import matplotlib.pyplot as pt


# We'll integrate
# 
# $$ y'=\alpha y$$
# 
# with $y'(0) = 1$,
# 
# using Backward Euler.

# Here are a few parameter settings that exhibit different situations that can occur:

# In[23]:


#alpha = -1; h = 0.1; final_t = 20
#alpha = -1; h = 1; final_t = 20
alpha = -1; h = 1.5; final_t = 20
#alpha = 1; h = 0.1; final_t = 20
#alpha = 1; h = 2; final_t = 20


# We specify the right-hand side and the initial condition:

# In[24]:


t_values = [0]
y_values = [1]

def f(y):
    return alpha * y


# Integrate in time using Forward Euler:

# In[25]:


while t_values[-1] < final_t:
    t_values.append(t_values[-1] + h)
    y_values.append(y_values[-1]/(1-h*alpha))


# And plot:

# In[26]:


mesh = np.linspace(0, final_t, 100)
pt.plot(t_values, y_values)
pt.plot(mesh, np.exp(alpha*mesh), label="true")
pt.legend()


# In[ ]:




