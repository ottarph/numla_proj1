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

    seeds = [1, 2, 4, 20]

    l_min, l_max = 0.01, 0.99

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    h = 4 / (np.sqrt(l_max) + np.sqrt(l_min))**2
    l = (np.sqrt(l_max) - np.sqrt(l_min)) / (np.sqrt(l_max) + np.sqrt(l_min))

    for seed in seeds:
        print(seed)
        L = random_test_matrix(n, seed=seed, off_diagonal=False, l_min=0.01, l_max=0.99)
        #L_0 = np.copy(L.todense())
        plt.figure()
 

        h = 0.7
        l = 0.7

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

        plt.title(f'seed $ = {seed} $')
        plt.legend()

    #print(L.todense() - L_0)

    
    plt.show()

    return

if __name__ == '__main__':
    main()
