#!/usr/bin/env python
# coding: utf-8

# In[1]:


import quadpy
import numpy as np


# In[ ]:





# In[4]:


scheme = quadpy.triangle.xiao_gimbutas_08()
scheme.show()
val = scheme.integrate(lambda x: np.exp(x[0]), [[0.0, 0.0], [1.0, 0.0], [0.5, 0.7]])
print(val)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'scheme.integrate')


# In[ ]:




