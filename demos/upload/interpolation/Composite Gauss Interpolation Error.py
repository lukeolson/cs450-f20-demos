#!/usr/bin/env python
# coding: utf-8

# In[8]:


from __future__ import division
import numpy as np
import scipy as sp
import scipy.special as ss
import matplotlib.pyplot as pt
import numpy.linalg as la

get_ipython().run_line_magic('pylab', 'inline')


# In[21]:


nelements = 5
nnodes = 3
mesh = np.linspace(-1, 1, nelements+1, endpoint=True)
gauss_nodes = ss.legendre(nnodes).weights[:, 0]*0.5 + 0.5

widths = np.diff(mesh)
nodes = mesh[:-1, np.newaxis] + widths[:, np.newaxis] * gauss_nodes


# In[22]:


def f(x):
    return np.abs(x-0.123812378)

pt.plot(nodes.flat, f(nodes).flat)


# In[23]:


nmany_nodes = 32

many_gauss_nodes = ss.legendre(nmany_nodes).weights[:, 0]*0.5 + 0.5
many_nodes = mesh[:-1, np.newaxis] + widths[:, np.newaxis] * many_gauss_nodes

def legendre_vdm(nodes, nmodes):
    result = np.empty((len(nodes), nmodes))
    for i in xrange(nmodes):
        result[:, i] = ss.eval_legendre(i, nodes)
    return result

vdm = legendre_vdm(gauss_nodes, nnodes)
many_vdm = legendre_vdm(many_gauss_nodes, nnodes)
zero_pad = np.zeros((nmany_nodes, nnodes))
zero_pad[:nnodes, :nnodes] = np.eye(nnodes)
upterpolate = np.dot(many_vdm, la.inv(vdm))


# In[24]:


fnodes = f(nodes)
fmany_nodes = np.dot(upterpolate, fnodes.T).T
pt.plot(many_nodes.flat, fmany_nodes.flat)


# In[ ]:




