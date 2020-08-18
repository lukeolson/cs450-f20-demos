#!/usr/bin/env python
# coding: utf-8

# #  Computing the Weights in Newton-Cotes Rules

# In[7]:


import numpy as np
import numpy.linalg as la


# We start by choosing our *quadrature nodes*, the maximum degree which will be exact, as well as the interval $(a,b)$ on which we integrate:

# In[16]:


#nodes = [0, 1]
#nodes = [0, 0.5, 1]
#nodes = [3, 3.5, 4]
#nodes = [0, 1, 2]
#nodes = np.linspace(0,1,5)
nodes = np.linspace(0, 1, 15)

max_degree = len(nodes)-1

a = nodes[0]
b = nodes[-1]


# Next, we compute the transpose of the Vandermonde matrix $V^T$ and the integrals $\int_a^b x^i$ as `rhs`:

# In[17]:


nodes = np.array(nodes)
powers = np.arange(max_degree+1)

Vt = nodes ** powers.reshape(-1, 1)

rhs = 1/(powers+1) * (b**(powers+1) - a**(powers+1))

if len(nodes) <= 4:
    print(Vt)
    print(rhs)


# Set up the linear system for the weights:
# 
# $$
# \begin{align*}
# \alpha_0 x_0^0 + \cdots + \alpha_{n-1} x_{n-1}^{0} &= \int_a^b x^0\\
# \vdots &= \vdots \\
# \alpha_0 x_0^{n-1} + \cdots + \alpha_{n-1} x_{n-1}^{n-1} &= \int_a^b x^{n-1}
# \end{align*}
# $$

# In[18]:


weights = la.solve(Vt, rhs)

print(weights)


# Now we test our quadrature rule by integrating the monomials $\int_a^b x^i dx$ and comparing quadrature results to the true answers:

# In[15]:


for i in range(len(nodes) + 2):
    approx = weights @ nodes**i
    
    true = 1/(i+1)*(b**(i+1) - a**(i+1))
    
    print("Error at degree %d: %g" % (i, approx-true))


# In[ ]:





# In[ ]:




