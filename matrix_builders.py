import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def build_L_sparse(n):

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
    if seed:
        np.random.seed(seed)

    X = np.random.rand(n,n)

    return householder(X)[0]
    

def random_spd(n, seed=False):
    if seed:
        np.random.seed(seed)
    
    eigvals = np.random.random(n)
    assert np.all(eigvals > 0)
    L = np.diag(eigvals)

    Q = random_orthogonal(n)

    return Q @ L @ Q.T

    
def random_test_matrix(n, seed=False):
    if seed:
        np.random.seed(seed)

    B = random_spd(n)
    I = np.eye(n)

    L = sp.sparse.dok_matrix((n**2,n**2))

    L[:n,:n] = B
    for i in range(1, n):
        L[(i-1)*n:i*n, i*n:(i+1)*n] = I
        L[i*n:(i+1)*n, i*n:(i+1)*n] = B
        L[i*n:(i+1)*n, (i-1)*n:i*n] = I

    return L


def main():

    np.random.seed(0)
    n = 3
    
    L = random_test_matrix(n, seed=1)

    print(np.round(L.todense(), 2))



    


    return

if __name__ == '__main__':
    main()
