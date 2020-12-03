#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


# In[3]:


def plot_vectors(V):
    base = [0 for v in V]
    x = [np.array(v).ravel()[0] for v in V]
    y = [np.array(v).ravel()[1] for v in V]
    plt.quiver(base, base, x, y, color = ['r', 'b'], angles = 'xy', scale=1.0, scale_units='xy')
    maxy = max([abs(i)+1 for i in y])
    maxx = max([abs(i)+1 for i in y])
    plt.ylim(-1*maxy, maxy)
    plt.xlim(-1*maxx, maxx)


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# x : \left(\begin{matrix}
#     0\\
#     1\\
# \end{matrix}\right) \hspace{25pt}
# A\cdot x : \left(\begin{matrix}
#     4\\
#     3\\
# \end{matrix}\right)$

# In[4]:


A = np.matrix([[3,4],[4,3]])
x = [[0],[1]]
Ax = A.dot(x)
plot_vectors([x, Ax])


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# x : \left(\begin{matrix}
#     1\\
#     0\\
# \end{matrix}\right) \hspace{25pt}
# A\cdot x : \left(\begin{matrix}
#     3\\
#     4\\
# \end{matrix}\right)$

# In[5]:


x = [[1],[0]]
Ax = A.dot(x)
plot_vectors([x, Ax])


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# x : \left(\begin{matrix}
#     -1\\
#     0.1\\
# \end{matrix}\right) \hspace{25pt}
# A\cdot x : \left(\begin{matrix}
#     -2.6\\
#     -3.7\\
# \end{matrix}\right)$

# In[6]:


x = np.array([[-1],[0.1]])
Ax = np.array(A.dot(x))
plot_vectors([x, Ax])


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# x : \left(\begin{matrix}
#     1\\
#     1\\
# \end{matrix}\right) \hspace{25pt}
# A\cdot x : \left(\begin{matrix}
#     7\\
#     7\\
# \end{matrix}\right)$

# In[7]:


x = np.array([[1],[1]])
Ax = np.array(A.dot(x))
plot_vectors([x, Ax])


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# x : \left(\begin{matrix}
#     1\\
#     -1\\
# \end{matrix}\right) \hspace{25pt}
# A\cdot x : \left(\begin{matrix}
#     -1\\
#     1\\
# \end{matrix}\right)$

# In[8]:


x = np.array([[1],[-1]])
Ax = np.array(A.dot(x))
plot_vectors([x, Ax])


# In[ ]:




