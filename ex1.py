import numpy as np
import scipy as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib

from polyak_itermethods import *
from sparseitermethods import *
from spectral_radius import *
from matrix_builders import *


def main():

    n = 10
    print(f'n = {n}, n^2 = {n**2}')

    TOL = 1e-7
    RTOL = TOL

    h = 1 / (n + 1)

    L = build_L_sparse(n)

    b = np.ones(n**2) * h**2
    x_0 = np.ones(n**2)

    #'''
    start = time()
    x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL)
    end = time()
    T_jac = end - start
    print('Jac ', i_jac, f'{T_jac*1e3:.2f} ms', np.linalg.norm(L @ x_jac - b))

    #'''

    #'''
    start = time()
    x_gs, i_gs, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0, tol=TOL, rtol=RTOL)
    end = time()
    T_gs = end - start
    print('GS ', i_gs, f'{T_gs*1e3:.2f} ms', np.linalg.norm(L @ x_gs - b))

    #'''
    w = 1.5
    start = time()
    x_sor, i_sor, r_sor = successive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
    end = time()
    T_sor = end - start
    print('SOR ', i_sor, f'{T_sor*1e3:.2f} ms', np.linalg.norm(L @ x_sor - b), w)

    iterations = np.array([i_jac, i_gs, i_sor])
    times = np.array([T_jac, T_gs, T_sor])

    print(times / iterations * 1000, 'ms')
    

    plt.figure()
    
    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'k:', label='JAC')
    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'k--', label='GS')
    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k-', label=rf'SOR, $\omega = {w}$')
    
    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Relative residuals')
    plt.xlim(0, np.amax(iterations))
    
    plt.figure()

    pos = [0.25, 0.5, 0.75]

    plt.bar(pos, times * 1000, width=0.1, color='black', alpha=0.6, linewidth=1, edgecolor='black')
    plt.xticks(pos, ['Jacobi', 'Gauss-Seidel', rf'SOR, $\omega = {w}$'])
    plt.ylabel('Running time / ms')

    plt.figure()

    ww = np.linspace(1.0, 1.5, 11)
    ww = np.append(ww, np.linspace(1.5 + (1.6-1.5)/10, 1.6, 10))
    ww = np.append(ww, np.linspace(1.6 + (2.0-1.6)/8, 2.0, 8))
    
    iters = np.zeros_like(ww)
    for i, w in enumerate(ww):
        _, it, _ = successive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
        iters[i] = it

    plt.plot(ww, iters, 'k.-', linewidth=0.7)
    plt.axhline(np.amin(iters), linewidth=0.5, linestyle='dashed', color='black')
    print(f'w_min = {ww[np.argmin(iters)]}, iters_min = {np.amin(iters)}')

    plt.xlim(np.amin(ww), np.amax(ww))
    plt.xlabel('$\omega$')
    plt.ylabel('Iterations')
    
    plt.show()

    return


if __name__ == '__main__':
    main()
