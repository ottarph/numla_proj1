import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
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
    

def random_spd(n, seed=False):
    '''
        Returns a random symmetric positive-definite matrix.
    '''
    if seed:
        np.random.seed(seed)
    
    eigvals = np.random.random(n)
    assert np.all(eigvals > 0)
    L = np.diag(eigvals)

    Q = random_orthogonal(n)

    return Q @ L @ Q.T

    
def random_test_matrix(n, seed=False):
    '''
        Though stated to in the project description, this method does not produce a positive,
        or negative, definite matrix.
    '''
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

    n = 10

    TOL = 1e-7
    RTOL = 1e-7



    L = random_test_matrix(n, seed=0)
    L = random_spd(n**2)

    L = sp.sparse.csc_matrix(L)
    L = random_test_matrix(n, seed=1) # 3
    print(np.amin(np.linalg.eigvals(L.todense())))

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 0.5
    l = 0.3

    ''' Jacobi with Polyak Heavy ball
    start = timer()
    x_jacHB, i_jacHB, r_jacHB = jacobi_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL)
    end = timer()
    print('JacHB ', i_jacHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jacHB - b))

    plt.semilogy(list(range(len(r_jacHB))), r_jacHB/r_jacHB[0], 'r-', label=rf'JacHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' Jacobi without
    start = timer()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL)
    end = timer()
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'r--', label='Jac, rel')
    #'''

    #print(np.linalg.eigvals(L.todense()))
    #print(jacobi_spectral_radius(L.todense()))

    h = 1.0
    l = 0.7

    ''' Forward Gauss-Seidel with Polyak Heavy ball
    start = timer()
    x_gsHB, i_gsHB, r_gsHB = f_gauss_seidel_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL)
    end = timer()
    print('GsHB  ', i_gsHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gsHB - b))

    plt.semilogy(list(range(len(r_gsHB))), r_gsHB/r_gsHB[0], 'b-', label=rf'GsHB, rel, $h = {h}, \lambda = {l}$')
    '''

    ''' Forward Gauss-Seidel without
    start = timer()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0, tol=TOL, rtol=RTOL)
    end = timer()
    print('Gs  ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))

    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'b--', label='Gs, rel')
    '''


    h = 1.25
    l = 0.75

    w = 1.5

    ''' SOR with Polyak Heavy ball
    start = timer()
    x_sorHB, i_sorHB, r_sorHB = successive_over_relaxation_heavy_ball_sparse(L, b, x_0, w, h, l, tol=TOL, rtol=RTOL)
    end = timer()
    print('SorHB ', i_sorHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sorHB - b))

    plt.semilogy(list(range(len(r_sorHB))), r_sorHB/r_sorHB[0], 'k-', label=rf'SorHB, rel, $h = {h}, \lambda = {l}$')
    '''

    ''' SOR without
    start = timer()
    x_sor, i_sor, r_sor = successive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
    end = timer()
    print('Sor ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sor - b))

    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k--', label='Sor, rel')
    '''

    h = 0.9
    l = 0.7

    ''' Max eigenvalue with Polyak Heavy ball
    start = timer()
    x_meHB, i_meHB, r_meHB = max_eigenvalue_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, Laplace=False)
    end = timer()
    print('MeHB ', i_meHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_meHB - b))

    plt.semilogy(list(range(len(r_meHB))), r_meHB/r_meHB[0], 'g-', label=rf'MeHB, rel, $h = {h}, \lambda = {l}$')
    '''

    ''' Max eigenvalue without
    start = timer()
    x_me, i_me, r_me = max_eigenvalue_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL, Laplace=False)
    end = timer()
    print('Me ', i_me, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_me - b))

    plt.semilogy(list(range(len(r_me))), r_me/r_me[0], 'g--', label=rf'Me, rel')
    '''

    

    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.title('Residuals')
    plt.xlabel('Iterations')
    #plt.xlim(0, max(i_sor, i_gs, i_jac))
    plt.show()

    return



    


    return

if __name__ == '__main__':
    main()
