#!/usr/bin/env python
# coding: utf-8

# # Writing Testable Numerics Code

# Here's the contents of a file containing numerics code:

# In[15]:


get_ipython().system('pygmentize norms.py')


# Note:
# 
# - Docstring
# - Defensive programming

# In[16]:


get_ipython().system('pygmentize test_norms.py')


# * Now use [pytest](https://pytest.org) to run the test.

# In[17]:


get_ipython().system('python -m pytest')


# A typical use for these tests would be to run them on every commit to a codebase.
# 
# Example: https://github.com/inducer/boxtree (click the "Pipeline" button)

# In[ ]:




