#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as pt


# In[12]:


from PIL import Image

with Image.open("house.jpeg").resize((500,500)) as img:
    rgb_img = np.array(img)
rgb_img.shape


# In[13]:


img = np.sum(rgb_img, axis=-1)


# In[14]:


pt.imshow(img, cmap="gray")


# In[15]:


u, sigma, vt = np.linalg.svd(img)
sigma

pt.plot(sigma)


# In[24]:



compressed_img = (
    sigma[0] * np.outer(u[:, 0], vt[0])
    + sigma[1] * np.outer(u[:, 1], vt[1])
    + sigma[2] * np.outer(u[:, 2], vt[2])
    + sigma[3] * np.outer(u[:, 3], vt[3])
    + sigma[4] * np.outer(u[:, 4], vt[4])
    + sigma[5] * np.outer(u[:, 5], vt[5])
    )

pt.imshow(compressed_img, cmap="gray")


# In[ ]:





# In[ ]:




