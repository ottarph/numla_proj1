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

    n = 10

    TOL = 1e-7
    RTOL = TOL


    L = build_L_sparse(n)

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 0.5
    l = 0.3

    #h = 1.24
    #l = 0.34

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


    L = random_spd(n**2, seed=1)

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
    x_meHB, i_meHB, r_meHB = max_eigenvalue_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, Laplace=False)
    end = time()
    print('MeHB ', i_meHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_meHB - b))
    plt.semilogy(list(range(len(r_meHB))), r_meHB/r_meHB[0], 'k-.', label=rf'Max-eigenvalue, $h = {h}, \lambda = {l}$')
    #'''
    #''' Max eigenvalue without
    start = time()
    x_me, i_me, r_me = max_eigenvalue_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL, Laplace=False)
    end = time()
    print('Me ', i_me, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_me - b))
    plt.semilogy(list(range(len(r_me))), r_me/r_me[0], 'k:', label=rf'Max-eigenvalue')
    #'''

    plt.axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)
    plt.legend()
    plt.ylabel('Relative residuals')
    plt.xlabel('Iterations')
    plt.xlim(0, max(i_me, i_jac, i_meHB, i_jacHB))



    ''' Jacobi HB optimal values.
    N = 5
    it_min = 1000
    l_min, h_min = 0, 0
    iterations = np.zeros((N,N))
    ll = np.linspace(0, 2, N)
    hh = np.linspace(0, 2, N)
    for i, l in enumerate(ll):
        print(i)
        for j, h in enumerate(hh):
            try:
                _, it, _ = jacobi_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, max_iter=300)
                iterations[i,j] = it
                if it < it_min:
                    l_min, h_min = l, h
            except:
                pass

    iterations = np.where(iterations != 0, iterations, np.ones_like(iterations) * 400)

    print(np.unravel_index(np.argmin(iterations, axis=None), iterations.shape))
    i,j = np.unravel_index(np.argmin(iterations, axis=None), iterations.shape)
    
    N = 11

    ll = np.linspace(l_min - 0.4, l_min + 0.4, N)
    hh = np.linspace(h_min - 0.4, h_min + 0.4, N)
    iterations = np.zeros((N,N))
    for i, l in enumerate(ll):
        print(i)
        for j, h in enumerate(hh):
            try:
                _, it, _ = jacobi_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, max_iter=300)
                iterations[i,j] = it
                if it < it_min:
                    l_min, h_min = l, h
                    it_min = it
            except:
                pass

    
    iterations = np.where(iterations != 0, iterations, np.ones_like(iterations) * 400)

    print(f'l_min = {l_min}, h_min = {h_min}')

    lll, hhh = np.meshgrid(ll, hh)
    
    

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(lll, hhh, iterations, rstride=1, cstride=1, cmap=cm.viridis)
    plt.xlabel("$l$")
    plt.ylabel("$h$")
    #'''
    



    plt.show()

    return


if __name__ == '__main__':
    main()
