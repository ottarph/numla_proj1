import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from sparseitermethods import *



def polyak_heavy_ball_iteration(H, x_0, A, b, h, l, tol, rtol):

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

        x_k = x
        p_k = p

        p = p_k - h * H @ (A @ x_k - b) - hl * p_k
        x = x_k + h * p

        r = np.linalg.norm(A @ x - b)
        residues.append(r)

    return x, i, np.array(residues)


def jacobi_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol)


def f_gauss_seidel_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol)


def succesive_over_relaxation_heavy_ball_sparse(A, b, x_0, w, h, l, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol)


def main():

    

    n = 10
    dx = 1 / (n + 1)

    L = build_L_sparse(n)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 0.5
    l = 0.3

    #''' Jacobi with Polyak Heavy ball
    start = timer()
    x_jacHB, i_jacHB, r_jacHB = jacobi_heavy_ball_sparse(L, b, x_0, h, l)
    end = timer()
    print('JacHB ', i_jacHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jacHB - b))

    plt.semilogy(list(range(len(r_jacHB))), r_jacHB/r_jacHB[0], 'r-', label=rf'JacHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' Jacobi without
    start = timer()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0)
    end = timer()
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'r--', label='Jac, rel')
    #'''

    h = 1.0
    l = 0.7

    #''' Forward Gauss-Seidel with Polyak Heavy ball
    start = timer()
    x_gsHB, i_gsHB, r_gsHB = f_gauss_seidel_heavy_ball_sparse(L, b, x_0, h, l)
    end = timer()
    print('GsHB  ', i_gsHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gsHB - b))

    plt.semilogy(list(range(len(r_gsHB))), r_gsHB/r_gsHB[0], 'b-', label=rf'GsHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' Forward Gauss-Seidel without
    start = timer()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0)
    end = timer()
    print('Gs  ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))

    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'b--', label='Gs, rel')
    #'''

# Change to SOR

    h = 1.3
    l = 0.7

    w = 1.5

    #''' SOR with Polyak Heavy ball
    start = timer()
    x_sorHB, i_sorHB, r_sorHB = succesive_over_relaxation_heavy_ball_sparse(L, b, x_0, w, h, l)
    end = timer()
    print('SorHB ', i_sorHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sorHB - b))

    plt.semilogy(list(range(len(r_sorHB))), r_sorHB/r_sorHB[0], 'k-', label=rf'SorHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' SOR without
    start = timer()
    x_sor, i_sor, r_sor = succesive_over_relaxation_sparse(L, b, x_0, w)
    end = timer()
    print('Sor ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sor - b))

    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k--', label='Sor, rel')
    #'''


    plt.axhline(y=1e-7, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.title('Residuals')
    plt.xlabel('Iterations')
    plt.xlim(0, max(i_sor, i_gs, i_jac))
    plt.show()

    return

if __name__ == '__main__':
    main()
