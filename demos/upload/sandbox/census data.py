#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk')

from io import StringIO

import scipy.optimize

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ctxt ="""
    1900     75.995
    1910     91.972
    1920    105.711
    1930    123.203
    1940    131.669
    1950    150.697
    1960    179.323
    1970    203.212
    1980    226.505
    1990    249.633
    2000    281.422
    2010    308.746
"""


# In[3]:


d = np.genfromtxt(StringIO(ctxt))
year = np.int32(d[:,0])
count = d[:,1]
print(year)
print(count)


# In[4]:


plt.plot(year, count, 'o')


# In[5]:


plt.figure()
plt.plot(year, count, 'o')

def f1(t, a, b, c):
    return a*np.exp(b*t)+c 
p1 = [10, 0.001, 100]

def f2(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))
p2 = [4000, 1.0, 0.001]

def f3(t, a, b, c):
    return a / (1+b*np.exp(-c*t))
p3 = [4000, 1.0, 0.001]

xyear = np.linspace(1900, 2100)

f = f1
p = p1
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))

f = f2
p = p2
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))

f = f3
p = p3
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))

count[-1] = 1.05 * 308.746
plt.figure()
plt.plot(year, count, 'o')

def f1(t, a, b, c):
    return a*np.exp(b*t)+c 
p1 = [10, 0.001, 100]

def f2(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))
p2 = [4000, 1.0, 0.001]

def f3(t, a, b, c):
    return a / (1+b*np.exp(-c*t))
p3 = [4000, 1.0, 0.001]

xyear = np.linspace(1900, 2100)

f = f1
p = p1
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))

f = f2
p = p2
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))

f = f3
p = p3
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)
xcount = f(xyear-1900, *popt)
plt.plot(xyear, xcount, '-')
plt.text(xyear[-1], xcount[-1], f'{xcount[-1]}', bbox=dict(facecolor='white'))


# In[ ]:


popt


# In[ ]:





# In[ ]:


f = f1
p = p1
popt, pcov = scipy.optimize.curve_fit(f, year-1900, count, p0=p)


# In[ ]:


popt


# In[ ]:


a, b, c = popt


# In[ ]:


a * np.exp(b * (year-1900)) + c


# In[ ]:




