import numpy as np
import scipy as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from polyak_itermethods import *
from sparseitermethods import *
from spectral_radius import *
from matrix_builders import *



def main():
    '''
        Convergence comparisons for accelerated and un-accelerated
        Jacobi and maximum eigenvalue methods
    '''

    n = 10

    TOL = 1e-7
    RTOL = TOL


    L = build_L_sparse(n)

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 0.5
    l = 0.3


    #''' Jacobi with Polyak Heavy ball
    start = time()
    x_jacHB, i_jacHB, r_jacHB = jacobi_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL)
    end = time()
    print('JacHB ', i_jacHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jacHB - b))

    plt.semilogy(list(range(len(r_jacHB))), r_jacHB/r_jacHB[0], 'k-', label=rf'Jacobi, $h = {h}, \lambda = {l}$')
    #'''

    #''' Jacobi without
    start = time()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL)
    end = time()
    print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'k--', label='Jacobi')
    #'''


    h = 0.9
    l = 0.7

    #''' Max eigenvalue with Polyak Heavy ball
    start = time()
    x_meHB, i_meHB, r_meHB = max_eigenvalue_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, Laplace=True)
    end = time()
    print('MeHB ', i_meHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_meHB - b))

    plt.semilogy(list(range(len(r_meHB))), r_meHB/r_meHB[0], 'k-.', label=rf'Max-eigenvalue, $h = {h}, \lambda = {l}$')
    #'''

    #''' Max eigenvalue without
    start = time()
    x_me, i_me, r_me = max_eigenvalue_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL, Laplace=True)
    end = time()
    print('Me ', i_me, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_me - b))

    plt.semilogy(list(range(len(r_me))), r_me/r_me[0], 'k:', label=rf'Max-eigenvalue')
    #'''

    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.ylabel('Relative residuals')
    plt.xlabel('Iterations')
    plt.xlim(0, max(i_me, i_jac, i_meHB, i_jacHB))



    plt.show()

    return


if __name__ == '__main__':
    main()
