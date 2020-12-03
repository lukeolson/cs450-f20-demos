#!/usr/bin/env python
# coding: utf-8

# # Polynomial fitting with the normal equations

# In[ ]:


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


# In this demo, we will produce data from a simple parabola as a "model" and try to recover the "unknown" parameters $\alpha$, $\beta$, and $\gamma$ using least squares.

# In[ ]:


alpha = 3
beta = 2
gamma = 2

def f(x):
    return alpha*x**2 + beta*x + gamma

plot_grid = np.linspace(-3, 3, 100)

pt.plot(plot_grid, f(plot_grid))


# In[ ]:


npts = 5

np.random.seed(22)
points = np.linspace(-2, 2, npts) + np.random.randn(npts)
values = f(points) + 0.3*np.random.randn(npts)*f(points)

pt.plot(plot_grid, f(plot_grid))
pt.plot(points, values, "o")


# Now build the Vandermonde matrix:

# In[ ]:


A = np.array([
    np.ones(npts),
    points,
    points**2
    ]).T
print(A)


# And solve for `x` using the normal equations:

# In[ ]:


x = la.solve(A.T@A, A.T@values)
x


# Lastly, pick apart `x` into `alpha_c`, `beta_c`, and `gamma_c`:

# In[ ]:


gamma_c, beta_c, alpha_c = x


# In[ ]:


print(alpha, alpha_c)
print(beta, beta_c)
print(gamma, gamma_c)


# In[ ]:


def f_c(x):
    return alpha_c*x**2 + beta_c*x + gamma_c

pt.plot(plot_grid, f(plot_grid), label="true")
pt.plot(points, values, "o", label="data")
pt.plot(plot_grid, f_c(plot_grid), label="found")
pt.legend()


# <!--
# gamma_c, beta_c, alpha_c = x
# -->
# (Edit this cell for solution.)
