import numpy as np
import scipy as sp
from time import time
import matplotlib.pyplot as plt
import matplotlib

from polyak_itermethods import *
from sparseitermethods import *
from spectral_radius import *
from matrix_builders import *


def beta(x, y):
    '''
        Estimator for the coefficient of beta in linear regression model
            y = alpha + beta * x
    '''
    n = x.shape[0]
    
    beta = np.sum( (x - np.mean(x)) * (y - np.mean(y))) / np.sum( (x - np.mean(x))**2 )

    return beta


def main():

    n = 10
    print(f'n = {n}, n^2 = {n**2}')

    TOL = 1e-7
    RTOL = TOL

    L = build_L_sparse(n)
    
    w = 1.57

    '''
        Finding the spectral radius of G for Jacobi, Gauss-Seidel and SOR methods
    '''
    rho_jac = jacobi_spectral_radius(L.todense())
    rho_gs = f_gauss_seidel_spectral_radius(L.todense())
    rho_sor = successive_over_relaxation_spectral_radius(L.todense(), w)

    print(f'Jacobi spectral radius = {rho_jac}')
    print(f'Gauss-Seidel spectral radius = {rho_gs}')
    print(f'SOR spectral radius = {rho_sor}')

    h = 1 / (n + 1)


    b = np.ones(n**2) * h**2
    x_0 = np.ones(n**2)

    '''
        Convergence comparison of Jacobi, Gauss-Seidel and SOR methods, TOL=1e-7
    '''

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

    
    iterationsA = np.array([i_jac, i_gs, i_sor])
    timesA = np.array([T_jac, T_gs, T_sor])

    print(f'Time per iteration, tolerance {TOL}', timesA / iterationsA * 1000, 'ms')
    
    plt.figure()
    
    plt.semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'k:', label='JAC')
    plt.semilogy(list(range(len(r_gs))), r_gs/r_gs[0], 'k--', label='GS')
    plt.semilogy(list(range(len(r_sor))), r_sor/r_sor[0], 'k-', label=rf'SOR, $\omega = {w}$')
    
    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Relative residuals')
    plt.xlim(0, np.amax(iterationsA))


    '''
        Convergence comparison of Jacobi, Gauss-Seidel and SOR methods, TOL=1e-7
    '''

    TOL = 1e-14
    RTOL = TOL
    
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

    iterationsB = np.array([i_jac, i_gs, i_sor])
    timesB = np.array([T_jac, T_gs, T_sor])

    print(f'Time per iteration, tolerance {TOL}', timesB / iterationsB * 1000, 'ms')

    plt.figure()

    pos = np.array([0.25, 0.5, 0.75])

    plt.bar(pos, timesA * 1000, width=0.05, color='black', alpha=0.4, linewidth=1, edgecolor='black', label=r'TOL $= 10^{-7}$')
    plt.bar(pos + 0.05, timesB * 1000, width=0.05, color='black', alpha=0.8, linewidth=1, edgecolor='black', label=r'TOL $= 10^{-14}$')
    plt.xticks(pos, ['Jacobi', 'Gauss-Seidel', rf'SOR, $\omega = {w}$'])
    plt.ylabel('Running time / ms')
    plt.legend()


    '''
        Finding optimal omega-value for SOR
    '''

    TOL = 1e-7
    RTOL = TOL
    ww = np.linspace(1.0, 1.5, 11)
    ww = np.append(ww, np.linspace(1.5 + (1.6-1.5)/10, 1.6, 10))
    ww = np.append(ww, np.linspace(1.6 + (2.0-1.6)/8, 2.0, 8))
    
    iters = np.zeros_like(ww)
    for i, w in enumerate(ww):
        _, it, _ = successive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
        iters[i] = it
    
    plt.figure()

    plt.plot(ww, iters, 'k.-', linewidth=0.7)
    plt.axhline(np.amin(iters), linewidth=0.5, linestyle='dashed', color='black')
    print(f'w_min = {ww[np.argmin(iters)]}, iters_min = {np.amin(iters)}')

    plt.xlim(np.amin(ww), np.amax(ww))
    plt.xlabel('$\omega$')
    plt.ylabel('Iterations')


    '''
        Numerical experiment for convergence results
    '''

    TOL = 1e-7
    RTOL = TOL

    _, _, r_jac = jacobi_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL)
    i_jac = np.arange(1, r_jac.shape[0] + 1, dtype=float)
    _, _, r_gs = f_gauss_seidel_sparse_fp(L, b, x_0, tol=TOL, rtol=RTOL)
    i_gs = np.arange(1, r_gs.shape[0] + 1, dtype=float)
    w = 1.57
    _, _, r_sor = successive_over_relaxation_sparse(L, b, x_0, w, tol=TOL, rtol=RTOL)
    i_sor = np.arange(1, r_sor.shape[0] + 1, dtype=float)

    print(f'Jacobi spectral radius = {rho_jac}')
    print(f'Gauss-Seidel spectral radius = {rho_gs}')
    print(f'SOR spectral radius = {rho_sor}')

    print(f'beta_jac = {np.exp(beta(i_jac, np.log(r_jac)))}')
    print(f'beta_gs = {np.exp(beta(i_gs, np.log(r_gs)))}')
    print(f'beta_sor (w = {w}) = {np.exp(beta(i_sor, np.log(r_sor)))}')

    
    plt.show()

    return


if __name__ == '__main__':
    main()
