#!/usr/bin/env python
# coding: utf-8

# Sympy variables are created using unique string identifiers.

# In[1]:


import sympy as sp
import numpy as np
x = sp.Symbol("x")
y = sp.Symbol("y")
z = sp.Symbol("z")


# One can form expression from symbols.
# Sympy expressions are made up of numbers, symbols, and sympy functions.

# In[2]:


expression = x**2. + y**2. + z ** 2.
expression


# Two expressions may be added together to form a new one.

# In[3]:


other_expression = x**2.
expression += other_expression
expression


# One can form sympy `Matrix` objects.

# In[4]:


sp.Matrix([[1,2],[3,4]])


# An important `Matrix` function is `eye(n)`, which forms a $n \times n$ identity matrix.

# In[5]:


sp.eye(3)


# One can stuff expressions into matrices, too.

# In[6]:


# One can stuff expressions into matrices
f1 = x**2.+y**2-z**2.
f2 = 2*x + y + z
function_matrix = sp.Matrix([f1,f2])
function_matrix


# One may compute the Jacobian of vector valued functions, too.

# In[7]:


function_matrix.jacobian([x,y,z]) # pass in a list of Sympy Symbols to take the Jacobian


# Sympy expressions can be evaluated by passing in a Python dictionary mapping Symbol `Symbol`s to specific values.

# In[8]:


x_val = 1.0
y_val = 2.0
z_val = 3.0
values={"x":x_val,"y":y_val,"z":z_val}
f1.subs(values)


# One can even valuate the Jacobian of functions.

# In[9]:


J_mat = function_matrix.jacobian([x,y,z]).subs(values)
J_mat


# To convert a Sympy `Matrix` into a Numpy array, one may use the following:

# In[10]:


np.array(J_mat)


# After evaluating an expression in Sympy, the return type is a `sympy.Float`.
# However, this is not readily usable by Numpy. Therefore, consider casting `sympy.Float` to a `numpy.float64`.

# In[11]:


J=np.array(J_mat).astype(np.float64)
J


# At this point, one cna do all the usual stuff one would in Numpy.

# In[12]:


J.T@J


# Symp's `Lambdify` can help increase the speed of Sympy's numerical computations.

# In[13]:


function_matrix.subs(values)


# In[14]:


from sympy.utilities.lambdify import lambdify
array2mat = [{'ImmutableDenseMatrix': np.array}, 'numpy']
lam_f_mat = lambdify((x,y,z), function_matrix, modules=array2mat)
lam_f_mat(1,2,3)


# Or if it is more convenient to have the function evaluation occur from a list of some sort, the Python `*` operator on lists can help.

# In[15]:


lam_f_mat(*[1,2,3])

