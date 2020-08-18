#!/usr/bin/env python
# coding: utf-8

# ### Conjugate Gradient: the Krylov subspace method for Convex Quadratic Optimization $\def\b#1{\boldsymbol{ #1}}\def\B#1{\boldsymbol{ #1}}$
# 
# Conjugate gradient can be thought of as a Krylov subspace method for the quadratic optimization problem $\min(f(\b x)$ where
# $$f(\b x)= \frac{1}{2}\b x^T\b A \b x + \b c^T \b x$$
# and $\b A$ is symmetric positive definite (this makes the problem convex).
# 
# In this demo, we compare the result of conjugate gradient to an explicitly constructed Krylov subspace.
# 
# We start by picking a random $\b A$ and $\b c$:

# In[17]:


import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt

n = 32
# make A a random SPD matrix
Q = la.qr(np.random.randn(n, n))[0]
A = Q @ (np.diag(np.random.rand(n)) @ Q.T)

c = np.random.randn(n)

# pick number of steps of CG to execute
k = 10

# define quadratic objective function
def f(x):
    return .5*np.inner(x, A @ x) + np.inner(c,x)


# ### Quadratic Programming using Arnoldi
# 
# First, we form a Krylov subspace using Arnoldi iteration on $\b A$ with starting guess $\b c$, then select the minima via
# $$\B x_k = -||\B c||_2\B Q_k \B T_k^{-1} \B e_1 ,$$
# which is a minimizes within this Krylov subspace since
# \begin{align*}
# \min_{\B x \in \mathcal{K}_k(\B A, \B c)} f(\B x) &= \min_{\B y \in \mathbb{R}^k} f(\B Q_k \B y)  \\
# &=\min_{\B y \in \mathbb{R}^k} \B y^T \B Q_k \B A \B Q_k \B y + \B c^T \B Q_k \B y \\
# &= \min_{\B y \in \mathbb{R}^k} \B y^T \B T_k \B y + ||\B c||_2\B e_1^T \B y.
# \end{align*}
# 

# In[18]:


# initialize Krylov subspace, taking first column as c/||c||_2
Q = np.zeros((n,k))
T = np.zeros((k,k))
x_kr = np.zeros((n,k))
Q[:,0] = c / la.norm(c)

for i in range(1,k+1):
    # perform Arnoldi iteration
    u = A @ Q[:,i-1]
    for j in range(0,i):
        T[j,i-1] = np.inner(Q[:,j],u)
        u = u - T[j,i-1]*Q[:,j]
    if i<k:
        T[i,i-1] = la.norm(u)
        Q[:,i] = u/T[i,i-1]
    # compute and store new best minimizer of f(x)
    if i>0:
        # compute approximate minima x_kr as -||\b c||_2\b Q \b T^{-1} \b e_1
        e1 = np.zeros(i) 
        e1[0] = la.norm(c)
        x_kr[:,i-1] = - Q[:,:i] @ la.solve(T[:i,:i],e1)
        print(f(x_kr[:,i-1]), "= min_{ x_kr in Krylov subspace of dimension ", i,"} f(x_kr)")


# ### Nonlinear Programming using Parallel Tangents for Quadratic Objective
# 
# Next, we implement Conjugate Gradient using the parallel tangents method, which is general to arbitrary nonlinear objective functions (but the convergence and optimality properties are only true for a convex quadratic objective function).
# 
# The parallel tangent method performs two line minimizations at each iteration:
#   * Perform a step of steepest descent to generate $\B{\hat{x}}_{k}$ from $\B x_k$.
#   * Generate $\B x_{k+1}$ by minimizing over the line passing through $\B x_{k-1}$ and $\B{\hat{x}}_{k}$.
# 
# To initialize the parallel tangents method, we use the two first approximate minima obtained from the Krylov subspace method above. Subsequent iterates are then reproduced exactly!

# In[28]:


# perform conjugate gradient by parallel tangents method
x_cg_pp = np.zeros((n, k))
x_cg_pp[:,0] = x_kr[:,0]
x_cg_pp[:,1] = x_kr[:,1]

# perform CG method via parallel tangents method
for i in range(2,k):
    x = x_cg_pp[:,i-1]

    # define function for steepest descent, based on current iterate x and grad_f(x)
    # can use local variables within function
    def f_1d(a):
        return f(x + a*(A@x+c))

    # solve for best steepest descent step using golden section search and form x_hat
    a = sopt.golden(f_1d)
    x_hat = x + a * (A@x+c)

    # define function for extrapolation, based on last iterate x_cg_pp[:,i-2] and x_hat (the solution of steepest descent)
    # can use local variables within function
    def g_1d(a):
        return f(x_hat + a*(x_hat - x_cg_pp[:,i-2]))
    # solve for best extrapolation using golden section search and update x
    a = sopt.golden(g_1d)
    x = x_hat + a*(x_hat - x_cg_pp[:,i-2])

    # store CG iterate as next column of x_cg_pp
    x_cg_pp[:,i] = x
    print(f(x), " = f( conjugate gradient parallel tangents iterate", i, ")")


# ### Quadratic Programming by Conjugate Gradient
# 
# The 1D line minimizations in the parallel tangents method can be solved analytically given a quadratic objective, so the parallel tangents implementation can be transformed to use a couple of matrix vector products in place of the two golden section searches.

# In[30]:


# perform CG method via matrix-vector products
x_cg_mv = np.zeros((n, k))
x_cg_mv[:,0] = x_kr[:,0]
x_cg_mv[:,1] = x_kr[:,1]

# this implementation mimics parallel tangents implementation,
# one of the matrix vector products can in fact be avoided
for i in range(2,k):
    x = x_cg_mv[:,i-1]

    # compute gradient
    g = A@x + c

    # compute explicit equation for optimal step size for steepest descent
    a = - np.inner(g,g)/np.inner(g, A@g)

    # update x
    x_hat = x + a * g

    # determine new optimization direction
    d = x_hat - x_cg_mv[:,i-2]

    # compute explicit equation for optimal step size in direction d = x_hat - x_{k-1}
    a = - np.inner(g,d)/np.inner(d, A@d)

    #update x
    x = x_hat + a * d

    # store CG iterate as next column of x_cg_mv
    x_cg_mv[:,i] = x
    print(f(x), " = f( conjugate gradient matrix vector product iterate", i, ")")


# It can be shown that each step of Conjugate gradient causes an update $\b s_k = \b x_{k+1} - \b x_k$ that is $\b A$-conjugate ($\b A$-orthogonal) to the previous, i.e. $\b s_i^T\b A \b s_{i-1}$:

# In[34]:


for i in range(2,k):
    print(np.inner(x_cg_mv[:,i]-x_cg_mv[:,i-1],A @ (x_cg_mv[:,i-1]-x_cg_mv[:,i-2])))


# In fact $\b s_i^T \b A \b s_j= 0$ if $i\neq j$.

# In[39]:


print((x_cg_mv[:,1:]-x_cg_mv[:,:-1]).T @ A @ (x_cg_mv[:,1:]-x_cg_mv[:,:-1]))


# This fact permits a further optimization of the conjugate gradient iteration, resulting in only 1 matrx-vector product (the resulting work-efficient method can be found in many textbooks, papers, and on wikipedia).

# In[ ]:




