import numpy as np
import scipy as sp
from time import time
import matplotlib.pyplot as plt

from sparseitermethods import *
from matrix_builders import *


def polyak_heavy_ball_iteration(H, x_0, A, b, h, l, tol, rtol, max_iter):

    hl = h * l

    p = np.zeros_like(x_0)
    p_k = p
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
        p_k = p

        p = p_k - h * H @ (A @ x_k - b) - hl * p_k
        x = x_k + h * p

        r = np.linalg.norm(A @ x - b)
        residues.append(r)

    return x, i, np.array(residues)


def jacobi_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.diags(1 / A.diagonal())
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def f_gauss_seidel_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def successive_over_relaxation_heavy_ball_sparse(A, b, x_0, w, h, l, tol=1e-7, rtol=1e-7, max_iter=1000, invprint=False):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = time()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = time()
    if invprint:
        print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def max_eigenvalue_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000, Laplace=False, invprint=False):

    if Laplace:
        n = int(np.sqrt(x_0.shape[0]))

        rr = np.arange(1, n+1)
        v1 = np.sin(np.pi / (n + 1) * rr)

        V = np.outer(v1, v1)

        v = V.flatten()
        v = v / np.linalg.norm(v)

        A_d = sp.sparse.diags(A.diagonal())

        s = 4 * np.cos(np.pi / (n+1))

        A_1 = A_d + s * np.outer(v,v)
        A_1 = sp.sparse.csc_matrix(A_1)

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

        start = time()
        A_1_inv = sp.sparse.linalg.inv(A_1)
        A_1_inv = sp.sparse.csc_matrix(A_1_inv)
        end = time()
        if invprint:
            print(f'{(end-start)*1e3:.2f}ms inversion')
    

    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def main():

    
    return

if __name__ == '__main__':
    main()
