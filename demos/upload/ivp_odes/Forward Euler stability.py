#!/usr/bin/env python
# coding: utf-8

# # Stability Experiments for Forward Euler

# In[3]:


import numpy as np
import matplotlib.pyplot as pt


# We'll integrate
# 
# $$ y'=\alpha y$$
# 
# with $y'(0) = 1$,
# 
# using forward Euler.

# Here are a few parameter settings that exhibit different situations that can occur:

# In[22]:


#alpha = 1; h = 0.1; final_t = 20
#alpha = -1; h = 0.1; final_t = 20
#alpha = -1; h = 1; final_t = 20
alpha = -1; h = 1.5; final_t = 20
#alpha = -1; h = 2; final_t = 20
#alpha = -1; h = 2.5; final_t = 20


# We specify the right-hand side and the initial condition:

# In[23]:


t_values = [0]
y_values = [1]

def f(y):
    return alpha * y


# Integrate in time using Forward Euler:

# In[24]:


while t_values[-1] < final_t:
    t_values.append(t_values[-1] + h)
    y_values.append(y_values[-1] + h*f(y_values[-1]))


# In[ ]:


while t < t_final:
    y = y + h * y * np.sin(3*t)
    t += h
    


# And plot:

# In[25]:


mesh = np.linspace(0, final_t, 100)
pt.plot(t_values, y_values)
pt.plot(mesh, np.exp(alpha*mesh), label="true")
pt.legend()


# In[27]:


mesh = np.linspace(0, final_t, 15)
pt.plot(mesh, y_values-np.exp(alpha*mesh), label="true")
pt.legend()


# In[ ]:




