"""Generate a diffusion stencil

Supports isotropic diffusion (FE,FD), anisotropic diffusion (FE, FD), and
rotated anisotropic diffusion (FD).

The stencils include redundancy to maintain readability for simple cases (e.g.
isotropic diffusion).

Construct sparse matrix from a local stencil
"""

import numpy as np
import scipy.sparse as sparse


def stencil_grid(S, grid, dtype=None, format='csr'):
    """Construct a sparse matrix form a local matrix stencil

    Parameters
    ----------
    S : ndarray
        matrix stencil stored in rank N array
    grid : tuple
        tuple containing the N grid dimensions
    dtype :
        data type of the result
    format : string
        sparse matrix format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    A : sparse matrix
        Sparse matrix which represents the operator given by applying
        stencil S at each vertex of a regular grid with given dimensions.

    Notes
    -----
    The grid vertices are enumerated as arange(prod(grid)).reshape(grid).
    This implies that the last grid dimension cycles fastest, while the
    first dimension cycles slowest.  For example, if grid=(2,3) then the
    grid vertices are ordered as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).

    This coincides with the ordering used by the NumPy functions
    ndenumerate() and mgrid().

    Examples
    --------
    >>> from pyamg.gallery import stencil_grid
    >>> stencil = [-1,2,-1]  # 1D Poisson stencil
    >>> grid = (5,)          # 1D grid with 5 vertices
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.,  0.],
            [-1.,  2., -1.,  0.,  0.],
            [ 0., -1.,  2., -1.,  0.],
            [ 0.,  0., -1.,  2., -1.],
            [ 0.,  0.,  0., -1.,  2.]])

    >>> stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]] # 2D Poisson stencil
    >>> grid = (3,3)                            # 2D grid with shape 3x3
    >>> A = stencil_grid(stencil, grid, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
            [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]])

    """

    S = np.asarray(S, dtype=dtype)
    grid = tuple(grid)

    if not (np.asarray(S.shape) % 2 == 1).all():
        raise ValueError('all stencil dimensions must be odd')

    if len(grid) != np.ndim(S):
        raise ValueError('stencil rank must equal number of grid dimensions')

    if min(grid) < 1:
        raise ValueError('grid dimensions must be positive')

    N_v = np.prod(grid)  # number of vertices in the mesh
    N_s = (S != 0).sum()    # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(grid)))[:-1]
    indices = tuple(i.copy() for i in S.nonzero())
    for i, s in zip(indices, S.shape):
        i -= s // 2
        #i = (i - s) // 2
        #i = i // 2
        # i = i - (s // 2)
    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = S[S != 0].repeat(N_v).reshape(N_s, N_v)

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data):
        diag = diag.reshape(grid)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(grid)
                s[n] = slice(0, i)
                diag[tuple(s)] = 0
            elif i < 0:
                s = [slice(None)]*len(grid)
                s[n] = slice(i, None)
                diag[tuple(s)] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]),
                            dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_matrix((data, diags),
                             shape=(N_v, N_v)).asformat(format)


def diff2d(nx, ny):
    stencil = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return stencil_grid(stencil, (nx, ny))


def diff3d(nx, ny, nz):
    stencil = np.array([[[0.,  0.,  0.],
                         [0., -1.,  0.],
                         [0.,  0.,  0.]],
                        [[0., -1.,  0.],
                         [-1.,  6., -1.],
                         [0., -1.,  0.]],
                        [[0.,  0.,  0.],
                         [0., -1.,  0.],
                         [0.,  0.,  0.]]])
    return stencil_grid(stencil, (nx, ny, nz))


if __name__ == '__main__':
    import scipy.sparse.linalg as la
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import time

    n = 40
    h = 1.0 / (n - 1)

    A = diff2d(n)

    # exact solution u = sin(pi x) * sin(pi y)
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    f = h**2 * 2 * np.pi**2\
        * np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())

    t = time.time()
    u = la.spsolve(A, f)
    t = time.time() - t
    print("time = %g" % t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u.reshape(X.shape),
                    rstride=1, cstride=1, cmap=plt.cm.jet, lw=0)

    plt.show()
