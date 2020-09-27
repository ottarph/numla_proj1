import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from time import time
import matplotlib.pyplot as plt

from matrix_builders import *

def jacobi_fp_sparse(A, b, x_0, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.diags(1 / A.diagonal())
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol, max_iter)


def f_gauss_seidel_sparse_fp(A, b, x_0, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol, max_iter)

def successive_over_relaxation_sparse(A, b, x_0, w=1.1, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc())
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol, max_iter)


def max_eigenvalue_fp_sparse(A, b, x_0, tol=1e-7, rtol=1e-7, max_iter=1000, Laplace=False, invprint=False):

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

        start = time()
        A_1_inv = -0.25 * ( np.eye(n**2) + s / (4 - s) * np.outer(v,v) )
        A_1_inv = sp.sparse.csc_matrix(A_1_inv)
        end = time()
        if invprint:
            print(f'{(end-start)*1e3:.2f}ms inversion')

    else:
        
        A_d = sp.sparse.diags(A.diagonal())
        Amd = A - A_d

        V = sp.sparse.linalg.eigs(Amd)

        ind = np.argmax(V[0])

        s, v = V[0][ind], V[1][:,ind]
        s, v = float(s), v.astype(float)

        A_1 = A_d + s * np.outer(v,v)
        A_1 = sp.sparse.csc_matrix(A_1)
        A_2 = A_1 - A

        start = time()
        A_1_inv = sp.sparse.linalg.inv(A_1)
        A_1_inv = sp.sparse.csc_matrix(A_1_inv)
        end = time()
        if invprint:
            print(f'{(end-start)*1e3:.2f}ms inversion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    return fp_iteration(G, f, x_0, A, b, tol, rtol, max_iter)


def fp_iteration(G, f, x_0, A, b, tol, rtol, max_iter):

    x = np.copy(x_0)
    x_k = x
    r0 = np.linalg.norm(A @ x_0 - b)

    r = r0

    residues = [r]

    i = 0
    while r > tol and r / r0 > rtol:
        i += 1
        if i > max_iter:
            raise Exception("Failed to converge within bounds.")

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
    b = np.ones(n)


    A_s = dok_matrix(A)

    #'''
    start = time()
    x_jac, i_jac = jacobi_fp_sparse(A_s, b, np.ones(n))
    end = time()
    #print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_jac))
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_jac - b))
    #'''

    #'''
    start = time()
    x_gs, i_gs = f_gauss_seidel_sparse_fp(A_s, b, np.ones(n))
    end = time()
    #print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_gs))
    print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_gs - b))
    #'''

    w = 1.95
    print(f'w = {w}')
    start = time()
    x_sor, i_sor = successive_over_relaxation_sparse(A_s, b, np.ones(n), w)
    end = time()
    #print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_sor))
    print('SOR ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(A_s @ x_sor - b))

    '''
    its = [successive_over_relaxation_sparse(A_s, b, np.ones(n), w)[1] for w in np.linspace(1.8,2.0,10)]
    
    plt.plot(np.linspace(1.8,2.0,10), its)
    plt.show()
    '''

    return





def main():


    return

if __name__ == '__main__':
    main()
