import numpy as np
import scipy as sp
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib

from polyak_itermethods import *
from sparseitermethods import *
from spectral_radius import *
from matrix_builders import *


def main():

    n = 10
    print(f'n = {n}, n^2 = {n**2}')

    RTOL = 1e-7

    h = 1 / (n + 1)

    L = build_L_sparse(n)

    b = np.ones(n**2) * h**2
    x_0 = np.ones(n**2)

    #'''
    start = timer()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0, rtol=RTOL, Laplace=True)
    end = timer()
    T_jac = end - start
    #print('Jac ', i_jac, f'{T_jac*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

    #'''

    #'''
    start = timer()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0, rtol=RTOL)
    end = timer()
    T_gs = end - start
    #print('GS ', i_gs, f'{T_gs*1e0:.2f} s', np.linalg.norm(L @ x_gs - b))

    #'''

    w = 1.5
    #w = 1.561
    #w = 2 / (1 + np.sin(np.pi / (n+1)))
    #print(f'w = {w}')
    start = timer()
    x_sor, i_sor, r_sor = successive_over_relaxation_sparse(L, b, x_0, w, rtol=RTOL)
    end = timer()
    T_sor = end - start
    #print('SOR ', i_sor, f'{T_sor*1e0:.2f} s', np.linalg.norm(L @ x_sor - b), w)
    

    plt.figure()
    
    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'k:', label='JAC')
    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'k--', label='GS')
    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k-', label=rf'SOR, $\omega = {w}$')
    
    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Relative residuals')
    plt.xlim(0, max(i_sor, i_gs, i_jac))
    
    plt.figure()

    pos = [0.25, 0.5, 0.75]
    times = np.array([T_jac, T_gs, T_sor])

    plt.bar(pos, times * 1000, width=0.15, color='black', alpha=0.5, linewidth=0.2, edgecolor='black')
    #plt.xticks(pos, [r'$T_{jac}$', r'$T_{gs}$', r'$T_{sor}$'])
    plt.xticks(pos, ['Jacobi', 'Gauss-Seidel', 'SOR'])
    plt.ylabel('Running time / ms')


    
    plt.show()

    return


if __name__ == '__main__':
    main()
