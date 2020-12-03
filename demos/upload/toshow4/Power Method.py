#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


# In[ ]:


def plot_vectors(V, EigVec):
    base = [0 for e in EigVec]
    vx = [np.array(e).ravel()[0] for e in EigVec]
    vy = [np.array(e).ravel()[1] for e in EigVec]
    plt.quiver(base, base, vx, vy, color = ['b', 'b'], angles = 'xy', scale=1.0, scale_units='xy')
    base = [0 for v in V]
    x = [np.array(v).ravel()[0] for v in V]
    y = [np.array(v).ravel()[1] for v in V]
    plt.quiver(base, base, x, y, color = 'r', angles = 'xy', scale=1.0, scale_units='xy')
    maxy = max([abs(i)+1 for i in y])
    maxx = max([abs(i)+1 for i in y])
    plt.ylim(-1*maxy, maxy)
    plt.xlim(-1*maxx, maxx)


# $A : \left(\begin{matrix}
#     3 & 4\\
#     4 & 3\\
# \end{matrix}\right) \hspace{25pt}
# v0 : \left(\begin{matrix}
#     1\\
#     1\\
# \end{matrix}\right) \hspace{25pt}
# v1 : \left(\begin{matrix}
#     1\\
#     -1\\
# \end{matrix}\right)$

# In[ ]:


A = np.matrix([[3,4],[4,3]])
v0 = [[1],[1]]
v1 = [[1],[-1]]
x = [[0],[1]]
for i in range(400):
    x = A.dot(x)
plot_vectors([x], [v0,v1])


# In[ ]:


A = np.matrix([[3,4],[4,3]])
v0 = [[1],[1]]
v1 = [[1],[-1]]
x = [[0],[1]]
for i in range(3):
    x = A.dot(x)
    x = x / np.linalg.norm(x,np.inf)
plot_vectors([x], [v0,v1])


# $A : \left(\begin{matrix}
#     2 & 0\\
#     0 & 2\\
# \end{matrix}\right) \hspace{25pt}
# v0 : \left(\begin{matrix}
#     1\\
#     0\\
# \end{matrix}\right) \hspace{25pt}
# v1 : \left(\begin{matrix}
#     0\\
#     1\\
# \end{matrix}\right)$

# In[ ]:


A = np.matrix([[2, 0],[0,2]])
v0 = [[1],[0]]
v1 = [[0],[1]]
x = [[1],[1]]
for i in range(100):
    x = A.dot(x)
    x = x / np.linalg.norm(x,np.inf)
plot_vectors([x], [v0,v1])


# In[ ]:





# In[ ]:




