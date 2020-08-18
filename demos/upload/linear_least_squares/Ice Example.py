#!/usr/bin/env python
# coding: utf-8

# http://ww2.amstat.org/publications/jse/v21n1/witt.pdf
# 
# http://nsidc.org/research/bios/fetterer.html
# 
# ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/monthly/data/N_08_extent_v3.0.csv

# In[19]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

from scipy import stats


# ## Read Data

# In[20]:


data = pd.read_csv('N_09_extent_v3.0.csv', dtype={'year': np.int32, 'extent': np.double})


# In[21]:


data


# In[22]:


data.dtypes


# In[30]:


year = data['year']
extent = data[' extent']


# In[31]:


plt.figure(figsize=(12,8))
plt.plot(year, extent, 'o')


# ## Try a linear fit

# In[40]:


year = data['year'][:-5]
extent = data[' extent'][:-5]
slope, intercept, rvalue, pvalue, stderr = stats.linregress(year, extent)
yfit = slope*year + intercept

plt.figure(figsize=(12,8))
plt.plot(year, yfit, 'r-')
plt.plot(year, extent, 'o')
print(slope)
print(intercept)
print(rvalue)
print(rvalue**2)


# ### How far off is this fit?

# In[41]:


plt.plot(year, extent - yfit, 'o')
slope
intercept


# ### How did the linear fit "fit" as time time progresses?

# In[42]:


plt.figure(figsize=(20,8))
res = []
for y in range(22, len(extent)+1):
    slope, intercept, rvalue, pvalue, stderr =     stats.linregress(year[:y], extent[:y])
    yfit = slope*year[:y] + intercept

    plt.plot(year[:y], yfit, '-', label='%d' % (1979+y))

plt.plot(year[:y], extent[:y], 'o')
plt.legend()


# ### Let's try a quadratic fit

# In[43]:


quadratic, linear, intercept = np.polyfit(year, extent, 2)
yfit = quadratic*year**2 + linear*year + intercept

plt.figure(figsize=(12,8))
plt.plot(year, yfit, 'r-')
plt.plot(year, extent, 'o')

rvalue = np.sqrt(1 - ((yfit - extent)**2).sum() / ((extent - np.mean(extent))**2).sum())

print(rvalue)


# In[44]:


plt.plot(year, extent - yfit, 'o')


# ## What about a cubic?
# 
# What goes wrong here?

# In[15]:


year = year
cubic, quadratic, linear, intercept = np.polyfit(year, extent, 3)
yfit = cubic*year**3 + quadratic*year**2 + linear*year + intercept

plt.figure(figsize=(12,8))
plt.plot(year, yfit, 'r-')
plt.plot(year, extent, 'o')

rvalue = np.sqrt(1 - ((yfit - extent)**2).sum() / ((extent - np.mean(extent))**2).sum())

print(rvalue)


# In[16]:


plt.plot(year, extent - yfit, 'o')


# In[ ]:




