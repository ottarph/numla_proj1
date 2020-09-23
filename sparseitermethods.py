import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from matrix_builders import *

def jacobi_fp_sparse(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol)


def f_gauss_seidel_sparse_fp(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol)

def succesive_over_relaxation_sparse(A, b, x_0, w=1.1, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc())
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol)


def max_eigenvalue_fp_sparse(A, b, x_0, tol=1e-7, rtol=1e-7, max_iter=1000, Laplace=False):

    if Laplace:
        n = int(np.sqrt(x_0.shape[0]))

        rr = np.arange(1, n+1)
        v1 = np.sin(np.pi / (n + 1) * rr)

        V = np.outer(v1, v1)

        v = V.flatten()
        v = v / np.linalg.norm(v)

        A_d = sp.sparse.diags(A.diagonal())

        u = (A - A_d) @ v

        s = 4 * np.cos(np.pi / (n+1))

        A_1 = A_d + s * np.outer(v,v)
        A_1 = sp.sparse.csc_matrix(A_1)
        A_2 = A_1 - A

        start = timer()
        A_1_inv = -0.25 * ( np.eye(n**2) + s / (4 - s) * np.outer(v,v) )
        A_1_inv = sp.sparse.csc_matrix(A_1_inv)
        end = timer()
        print(f'{(end-start)*1e3:.2f}ms inversion')

    else:
        raise NotImplementedError

    G = A_1_inv @ A_2
    f = A_1_inv @ b
    #print(G, f)

    return fp_iteration(G, f, x_0, A, b, tol, rtol)


def elementwise_jacobi(A, b, x_0, tol=1e-7, rtol=1e-7):
    
    
    return


def fp_iteration(G, f, x_0, A, b, tol, rtol):

    x = np.copy(x_0)
    x_k = x
    r0 = np.linalg.norm(A @ x_0 - b)

    r = r0

    residues = [r]

    i = 0
    while r > tol and r / r0 > rtol:
        i += 1

        x_k = x
        x = G @ x_k + f
        r = np.linalg.norm(A @ x - b)
        residues.append(r)

    return x, i, np.array(residues)


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




    

def main():

    n = 10
    print(f'n = {n}, n^2 = {n**2}')

    h = 1 / (n + 1)

    L = build_L_sparse(n)

    b = np.ones(n**2) * h**2
    x_0 = np.ones(n**2)

    #'''
    start = timer()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0)
    end = timer()
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

    plt.semilogy(list(range(len(r_jac))), r_jac, 'r-', label='JAC')
    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'r--', label='JAC, rel')
    #'''

    #'''
    start = timer()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0)
    end = timer()
    print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))

    plt.semilogy(list(range(len(r_gs))), r_gs, 'b-', label='GS')
    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'b--', label='GS, rel')

    #'''

    w = 1.5
    w = 1.561
    w = 2 / (1 + np.sin(np.pi / (n+1)))
    print(w)
    print(f'w = {w}')
    start = timer()
    x_sor, i_sor, r_sor = succesive_over_relaxation_sparse(L, b, x_0, w)
    end = timer()
    print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sor - b))
    
    plt.semilogy(list(range(len(r_sor))), r_sor, 'k-', label=rf'SOR, $\omega = {w}$')
    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k--', label='SOR, rel')
    
    plt.axhline(y=1e-7, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.title('Residuals')
    plt.xlabel('Iterations')
    plt.xlim(0, max(i_sor, i_gs, i_jac))
    plt.show()



    return

if __name__ == '__main__':
    main()
