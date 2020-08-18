#!/usr/bin/env python
# coding: utf-8

# # LU with Partial Pivoting

# In[131]:


import numpy as np
import numpy.linalg as la

np.set_printoptions(precision=3, suppress=True)


# # Set-up

# Let's grab a (admittedly well-chosen) sample matrix `A`:

# In[132]:


n = 4

np.random.seed(235)
A = np.round(5*np.random.randn(n, n))
A[0,0] = 0
A[2,1] = 17
A[0,2] = 19
A


# ----------------
# ## Permutation matrices
# 
# Now define a function `row_swap_mat(i, j)` that returns a permutation matrix that swaps row i and j:

# In[133]:


def row_swap_mat(i, j):
    P = np.eye(n)
    P[i] = 0
    P[j] = 0
    P[i, j] = 1
    P[j, i] = 1
    return P


# What do these matrices look like?

# In[134]:


row_swap_mat(0,1)


# Do they work?

# In[135]:


row_swap_mat(0,1).dot(A)


# --------------

# # Part I

# `U` is the copy of `A` that we'll modify:

# In[136]:


U = A.copy()


# ## First column

# Create P1 to swap up the right row: 

# In[137]:


P1 = row_swap_mat(0, 3)
U = P1.dot(U)
U


# In[138]:


M1 = np.eye(n)
M1[1,0] = -U[1,0]/U[0,0]
M1[2,0] = -U[2,0]/U[0,0]
M1


# In[139]:


U = M1.dot(U)
U


# ## Second column
# 
# Create `P2` to swap up the right row:

# In[140]:


P2 = row_swap_mat(2,1)
U = P2.dot(U)
U


# Make the second-column elimination matrix `M2`:

# In[141]:


M2 = np.eye(n)
M2[2,1] = -U[2,1]/U[1,1]
M2[3,1] = -U[3,1]/U[1,1]
M2


# In[142]:


U = M2.dot(U)
U


# ## Third column
# 
# Create `P3` to swap up the right entry:

# In[143]:


P3 = row_swap_mat(3, 2)
U = P3.dot(U)
U


# Make the third-column elimination matrix `M3`:

# In[144]:


M3 = np.eye(n)
M3[3,2] = -U[3,2]/U[2,2]
M3


# In[145]:


U = M3.dot(U)
U


# ## Wrap-up

# So we've built $M3P_3M_2P_2M_1P_1A=U$.

# In[150]:


M3.dot(P3).dot(M2).dot(P2).dot(M1).dot(P1).dot(A)


# ---------------------
# That left factor is anything but lower triangular:

# In[151]:


M3.dot(P3).dot(M2).dot(P2).dot(M1).dot(P1)


# # Part II

# Now try the reordering trick:

# In[160]:


L3 = M3
L2 = P3.dot(M2).dot(la.inv(P3))
L1 = P3.dot(P2).dot(M1).dot(la.inv(P2)).dot(la.inv(P3))


# In[155]:


L3.dot(L2).dot(L1).dot(P3).dot(P2).dot(P1)


# --------------
# We were promised that all of the `L`*n* are still lower-triangular:

# In[168]:


print(L1)
print(L2)
print(L3)


# So their product is, too:

# In[172]:


Ltemp = L3.dot(L2).dot(L1)
Ltemp


# ----
# `P` is still a permutation matrix (but a more complicated one):

# In[174]:


P = P3.dot(P2).dot(P1)
P


# -----------------
# So to sum up, we've made:

# In[175]:


Ltemp.dot(P).dot(A) - U


# --------------
# Multiply from the left by `Ltemp`${}^{-1}$, which is *also* lower triangular:

# In[179]:


L = la.inv(Ltemp)
L


# In[180]:


P.dot(A) - L.dot(U)

