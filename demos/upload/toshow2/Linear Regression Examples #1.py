#!/usr/bin/env python
# coding: utf-8

# Much of this is from the following source: https://onlinecourses.science.psu.edu/stat501/node/257

# In[1]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

from scipy import stats


# ### Example 1: building stories vs height

# In[2]:


data = pd.read_csv('./bldgstories.txt', delim_whitespace=True)


# In[3]:


data

year = data.values[:,0]
hght = data.values[:,1]
stories= data.values[:,2]


# In[4]:


plt.plot(hght, stories, 'o')
plt.xlabel('height')
plt.ylabel('stories')


# #### Add a linear regresssion line
# 
# This is of the form
# $$
# m x + b
# $$
# where $m$ is the slope and $b$ is the intercept.

# In[5]:


slope, intercept, rvalue, pvalue, stderr = stats.linregress(hght, stories)
yfit = slope*hght + intercept
plt.plot(hght, yfit, 'r-')

plt.plot(hght, stories, 'o')
plt.xlabel('height')
plt.ylabel('stories')


# #### Now add a line representing the distance to the linear line

# In[6]:


slope, intercept, rvalue, pvalue, stderr = stats.linregress(hght, stories)
yfit = slope*hght + intercept
plt.plot(hght, yfit, 'r-')

for i in range(len(hght)):
    x = hght[i]
    y = yfit[i]
    s = stories[i]
    plt.plot([x, x], [y, s], 'k-')

plt.plot(hght, stories, 'o')
plt.xlabel('height')
plt.ylabel('stories')


# ### How far away is each dot from the red line?
# 
# `yfit` is the straight line
# 
# `stories` is the data
# 
# If `yfit`-`stories` were 0, then this would be a perfect fit, meaning a strong correlation and a good fit with a line would be 100% (or 1.0).  In addition we divide by `stories`-mean(`stories`).

# In[7]:


np.sqrt(1 - ((yfit - stories)**2).sum() / ((stories - np.mean(stories))**2).sum())


# This is also called the rvalue or correlation coefficient.

# In[8]:


rvalue


# ### Example 2, eye sight distance vs age

# In[9]:


data = pd.read_csv('./signdist.txt', delim_whitespace=True)


# In[10]:


print(data)
age = data.values[:,0]
distance = data.values[:,1]


# In[11]:


slope, intercept, rvalue, pvalue, stderr = stats.linregress(age, distance)
yfit = slope*age + intercept
plt.plot(age, yfit, 'r-')


for i in range(len(age)):
    x = age[i]
    y = yfit[i]
    s = distance[i]
    plt.plot([x, x], [y, s], 'k-')

plt.plot(age, distance, 'o')
plt.xlabel('age')
plt.ylabel('distance')


# In[12]:


print(rvalue)


# ### Example 3, height vs GPA
# 
# What?!

# In[13]:


data = pd.read_csv('./heightgpa.txt', delim_whitespace=True)


# In[14]:


height = data.values[:,0]
gpa = data.values[:,1]


# In[15]:


slope, intercept, rvalue, pvalue, stderr = stats.linregress(height, gpa)
yfit = slope*height + intercept
plt.plot(height, yfit, 'r-')


for i in range(len(height)):
    x = height[i]
    y = yfit[i]
    s = gpa[i]
    plt.plot([x, x], [y, s], 'k-')

plt.plot(height, gpa, 'o')
plt.xlabel('height')
plt.ylabel('gpa')


# In[ ]:


print(rvalue)


# In[ ]:




