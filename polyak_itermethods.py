import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer
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


def jacobi_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def f_gauss_seidel_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000):

    A_1 = sp.sparse.tril(A)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def succesive_over_relaxation_heavy_ball_sparse(A, b, x_0, w, h, l, tol=1e-7, rtol=1e-7, max_iter=1000):

    A_1 = sp.sparse.diags(A.diagonal()) + w * sp.sparse.tril(A, k=-1)
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms inversion')


    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def max_eigenvalue_heavy_ball_sparse(A, b, x_0, h, l, tol=1e-7, rtol=1e-7, max_iter=1000, Laplace=False):

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

        start = timer()
        A_1_inv = -0.25 * ( np.eye(n**2) + s / (4 - s) * np.outer(v,v) )
        A_1_inv = sp.sparse.csc_matrix(A_1_inv)
        end = timer()
        print(f'{(end-start)*1e3:.2f}ms inversion')


    else:
        raise NotImplementedError

    

    return polyak_heavy_ball_iteration(A_1_inv, x_0, A, b, h, l, tol, rtol, max_iter)


def main():

    n = 10

    TOL = 1e-14
    RTOL = 1e-14



    L = build_L_sparse(n)

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 0.5
    l = 0.3

    #''' Jacobi with Polyak Heavy ball
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

    h = 1.0
    l = 0.7

    #''' Forward Gauss-Seidel with Polyak Heavy ball
    start = timer()
    x_gsHB, i_gsHB, r_gsHB = f_gauss_seidel_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL)
    end = timer()
    print('GsHB  ', i_gsHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gsHB - b))

    plt.semilogy(list(range(len(r_gsHB))), r_gsHB/r_gsHB[0], 'b-', label=rf'GsHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' Forward Gauss-Seidel without
    start = timer()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0, tol=TOL, rtol=RTOL)
    end = timer()
    print('Gs  ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))

    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'b--', label='Gs, rel')
    #'''


    h = 1.25
    l = 0.75

    w = 1.5

    #''' SOR with Polyak Heavy ball
    start = timer()
    x_sorHB, i_sorHB, r_sorHB = succesive_over_relaxation_heavy_ball_sparse(L, b, x_0, w, h, l, tol=TOL, rtol=RTOL)
    end = timer()
    print('SorHB ', i_sorHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sorHB - b))

    plt.semilogy(list(range(len(r_sorHB))), r_sorHB/r_sorHB[0], 'k-', label=rf'SorHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' SOR without
    start = timer()
    x_sor, i_sor, r_sor = succesive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
    end = timer()
    print('Sor ', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_sor - b))

    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k--', label='Sor, rel')
    #'''

#region
    '''
    i_min = 100
    h_min, l_min = -1, -1
    for h in np.linspace(1.2, 1.3, 10):
        print(f'h = {h}')
        for l in np.linspace(0.7, 0.8, 10):
            try:
                x, i, r = succesive_over_relaxation_heavy_ball_sparse(L, b, x_0, w, h, l, max_iter=100)
                if i < i_min:
                    i_min = i
                    h_min, l_min = h, l
            except:
                pass
    print(i_min, h_min, l_min)
    '''
#endregion

    h = 0.9
    l = 0.7

    #''' Max eigenvalue with Polyak Heavy ball
    start = timer()
    x_meHB, i_meHB, r_meHB = max_eigenvalue_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, Laplace=True)
    end = timer()
    print('MeHB ', i_meHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_meHB - b))

    plt.semilogy(list(range(len(r_meHB))), r_meHB/r_meHB[0], 'g-', label=rf'MeHB, rel, $h = {h}, \lambda = {l}$')
    #'''

    #''' Max eigenvalue without
    start = timer()
    x_me, i_me, r_me = max_eigenvalue_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL, Laplace=True)
    end = timer()
    print('Me ', i_me, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_me - b))

    plt.semilogy(list(range(len(r_me))), r_me/r_me[0], 'g--', label=rf'Me, rel')
    #'''


    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.title('Residuals')
    plt.xlabel('Iterations')
    plt.xlim(0, max(i_sor, i_gs, i_jac))
    plt.show()

    return

if __name__ == '__main__':
    main()
