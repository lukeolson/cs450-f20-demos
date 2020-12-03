#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
faces = datasets.fetch_olivetti_faces()
imshape = np.shape(faces.images[0])
imsize = imshape[0]*imshape[1]


# ## We can analyze the pictures of a large variety of people:

# In[5]:


fig = plt.figure(figsize=(15,10))
for i in range(15):
    fig.add_subplot(3, 5, (i+1), xticks=[], yticks=[])
    plt.imshow(faces.images[i], cmap=plt.cm.gray)


# ## Actually, 400 different images:

# In[6]:


fig = plt.figure(figsize=(12,12))
for i in range(400):
    fig.add_subplot(20, 20, (i+1), xticks=[], yticks=[])
    plt.imshow(faces.images[i], cmap=plt.cm.gray)


# ## Of these 40 people, what is the average face?

# In[7]:


FaceMat = np.matrix([image.ravel() for image in faces.images])
AvgFace = np.mean(FaceMat, axis=0)
plt.imshow(AvgFace.reshape(imshape), cmap=plt.cm.gray)


# ## Using eigenvalues and eigenvectors, we can highlight a set of features that represent the variation among these faces

# In[8]:


FaceDiff = FaceMat - AvgFace
FaceCov = np.dot(FaceDiff.T, FaceDiff)
[vals, vecs] = np.linalg.eigh(FaceCov)
sort_idx = np.argsort(-vals)
vals = vals[sort_idx]
vecs = vecs[:, sort_idx]
def preserve_var(vals, variance = .95):
    for idx, cumsum in enumerate(np.cumsum(vals) / np.sum(vals)):
        if cumsum > variance:
            return idx
n_var = preserve_var(vals)
vals = vals[:n_var]
vecs = vecs[:, :n_var]
FaceEig = []
for i in range(vecs.shape[1]):
    FaceEig.append(np.asarray(vecs[:,i].reshape(imshape)))
fig = plt.figure(figsize=(20,10))
for i in range(20):
    fig.add_subplot(5, 5, (i+1), xticks=[], yticks=[])
    plt.imshow(np.reshape(FaceEig[i], imshape), cmap='jet')


# In[ ]:




