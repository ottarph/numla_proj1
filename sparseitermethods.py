import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def jacobi_fp_sparse(A, b, x_0, tol=1e-6, rtol=1e-6):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol)


def f_gauss_seidel_sparse_fp(A, b, x_0, tol=1e-6, rtol=1e-6):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol)

def succesive_over_relaxation_sparse(A, b, x_0, w=1.1, tol=1e-6, rtol=1e-6):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc())
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    #G = G.tocsr()

    return fp_iteration(G, f, x_0, A, b, tol, rtol)


def fp_iteration(G, f, x_0, A, b, tol, rtol):

    x = np.copy(x_0)
    x_k = x
    r0 = np.linalg.norm(A @ x_0 - b)

    r = r0

    i = 0
    while r > tol and r / r0 > rtol:
        i += 1

        x_k = x
        x = G @ x_k + f
        r = np.linalg.norm(A @ x - b)

    return x, i


def onedtest(n):
    A =  np.diag(np.full( n ,  2, dtype=float),  0)
    A += np.diag(np.full(n-1, -1, dtype=float), -1)
    A += np.diag(np.full(n-1, -1, dtype=float), +1)

    x = np.arange(1, n+1, dtype=float)
    #b = A @ x
    b = np.ones(n)


    A_s = dok_matrix(A)

    #'''
    start = timer()
    x_jac, i_jac = jacobi_fp_sparse(A_s, b, np.ones(n))
    end = timer()
    #print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_jac))
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_jac - b))
    #'''

    #'''
    start = timer()
    x_gs, i_gs = f_gauss_seidel_sparse_fp(A_s, b, np.ones(n))
    end = timer()
    #print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_gs))
    print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_gs - b))
    #'''

    w = 1.95
    print(f'w = {w}')
    start = timer()
    x_sor, i_sor = succesive_over_relaxation_sparse(A_s, b, np.ones(n), w)
    end = timer()
    #print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_sor))
    print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_sor - b))

    '''
    its = [succesive_over_relaxation_sparse(A_s, b, np.ones(n), w)[1] for w in np.linspace(1.8,2.0,10)]
    
    plt.plot(np.linspace(1.8,2.0,10), its)
    plt.show()
    '''

    return


def build_L(n):

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

    

def main():

    n = 20

    h = 1 / (n + 1)

    L = build_L(n)

    b = np.ones(n**2) * h**2
    x_0 = np.ones(n**2)

    #'''
    start = timer()
    x_jac, i_jac = jacobi_fp_sparse(L, b, x_0)
    end = timer()
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))
    #'''

    #'''
    start = timer()
    x_gs, i_gs = f_gauss_seidel_sparse_fp(L, b, x_0)
    end = timer()
    print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))
    #'''

    w = 1.5
    print(f'w = {w}')
    start = timer()
    x_sor, i_sor = succesive_over_relaxation_sparse(L, b, x_0, w)
    end = timer()
    print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sor - b))



    return

if __name__ == '__main__':
    main()
