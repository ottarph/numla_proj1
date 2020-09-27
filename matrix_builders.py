import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from time import time
import matplotlib.pyplot as plt

from polyak_itermethods import *
from sparseitermethods import *
from spectral_radius import *


def build_L_sparse(n):
    '''
        Builds the block-tridiagonal discrete Laplacian as a sparse matrix.
    '''

    B  = sp.sparse.diags(np.full( n , -4),  0)
    B += sp.sparse.diags(np.full(n-1,  1), -1)
    B += sp.sparse.diags(np.full(n-1,  1), +1)

    I = sp.sparse.identity(n)

    L = sp.sparse.dok_matrix((n**2,n**2))

    L[:n,:n] = B
    for i in range(1, n):
        L[(i-1)*n:i*n, i*n:(i+1)*n] = I
        L[i*n:(i+1)*n, i*n:(i+1)*n] = B
        L[i*n:(i+1)*n, (i-1)*n:i*n] = I

    return L


def householder(X):
    '''
        QR-factorization with householder projections.
    '''
    X = np.copy(X)
    n, m = X.shape

    Q = np.zeros((n,m))
    R = np.zeros((n,m))
    W = np.zeros((n,m))

    for k in range(m):
        rk = X[:,k]

        for i in range(k):
            rk -= 2 * np.inner(W[:,i], rk) * W[:,i]
        
        beta = np.sign(X[k,k]) * np.linalg.norm(X[k:,k])
        z = np.zeros(n)
        z[k:] += X[k:,k]
        z[k] += beta

        W[:,k] = z / np.linalg.norm(z)

        rk -= 2 * np.inner(W[:,k], rk) * W[:,k]

        R[:,k] = rk

        qk = np.zeros(n)
        qk[k] = 1
        for i in range(k, -1, -1):
            qk -= 2 * np.inner(W[:,i], qk) * W[:,i]
        Q[:,k] = qk

    return Q, R[:m,:m]

def random_orthogonal(n, seed=False):
    '''
        Builds a random orthogonal matrix by use of the Q from a QR-factorization
        of a random matrix. Since singular-matrices have measure zero, the
        algorithm is highly likely, although not guaranteed, to succeed.
    '''
    if seed:
        np.random.seed(seed)

    X = np.random.rand(n,n)

    Q = householder(X)[0]
    assert np.allclose(Q.T @ Q, np.eye(n))

    return Q
    

def random_spd(n, seed=False, l_min=0, l_max=1):
    '''
        Returns a random symmetric positive-definite matrix.
    '''
    if seed:
        np.random.seed(seed)
    
    eigvals = np.random.random(n) * (l_max - l_min) + l_min
    assert np.all(eigvals > 0)
    L = np.diag(eigvals)

    Q = random_orthogonal(n)

    return Q @ L @ Q.T

    
def random_test_matrix(n, seed=False, off_diagonal=True, l_min=0, l_max=1):
    '''
        Though stated to in the project description, this method does not produce a positive,
        or negative, definite matrix.
    '''
    if seed:
        np.random.seed(seed)

    B = random_spd(n, l_min=l_min, l_max=l_max)
    I = np.eye(n)

    L = sp.sparse.dok_matrix((n**2,n**2))

    L[:n,:n] = B
    if off_diagonal:
        for i in range(1, n):
            L[(i-1)*n:i*n, i*n:(i+1)*n] = -I
            L[i*n:(i+1)*n, i*n:(i+1)*n] = B
            L[i*n:(i+1)*n, (i-1)*n:i*n] = -I
    else:
        for i in range(1, n):
            L[i*n:(i+1)*n, i*n:(i+1)*n] = B

    return L


def main():

    
    return


if __name__ == '__main__':
    main()
