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

    l_min, l_max = 0.00, 0.99

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)
    
    fig, axs = plt.subplots(2, len(seeds) // 2)

    jac_radii = np.zeros_like(seeds)
    max_eig_radii = np.zeros_like(seeds)

    matplotlib.rcParams.update({'font.size': 16})

    for i, seed in enumerate(seeds):
        print(seed)
        L = random_test_matrix(n, seed=seed, off_diagonal=False, l_min=0.01, l_max=0.99)

        h = 1.24
        l = 0.34

        #''' Jacobi with Polyak Heavy ball
        start = time()

        x_jacHB, i_jacHB, r_jacHB = jacobi_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL)
        end = time()
        print('JacHB ', i_jacHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jacHB - b))

        axs.flatten()[i].semilogy(list(range(len(r_jacHB))), r_jacHB/r_jacHB[0], 'k-', label=rf'Jacobi, $h = {h}, \lambda = {l}$')
        #'''

        #''' Jacobi without
        start = time()
        x_jac, i_jac, r_jac = jacobi_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL)
        end = time()
        print('Jac ', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_jac - b))

        axs.flatten()[i].semilogy(list(range(len(r_jac))), r_jac/r_jac[0], 'k--', label='Jacobi')
        #'''

        h = 0.9
        l = 0.7

        #''' Max eigenvalue with Polyak Heavy ball
        start = time()
        x_meHB, i_meHB, r_meHB = max_eigenvalue_heavy_ball_sparse(L, b, x_0, h, l, tol=TOL, rtol=RTOL, Laplace=False)
        end = time()
        print('MeHB ', i_meHB, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_meHB - b))

        axs.flatten()[i].semilogy(list(range(len(r_meHB))), r_meHB/r_meHB[0], 'k-.', label=rf'Max-eigenvalue, $h = {h}, \lambda = {l}$')
        #'''

        #''' Max eigenvalue without
        start = time()
        x_me, i_me, r_me = max_eigenvalue_fp_sparse(L, b, x_0, tol=TOL, rtol=RTOL, Laplace=False)
        end = time()
        print('Me ', i_me, f'{(end - start)*1e0:.2f} s', np.linalg.norm(L @ x_me - b))

        axs.flatten()[i].semilogy(list(range(len(r_me))), r_me/r_me[0], 'k:', label=rf'Max-eigenvalue')
        #'''

        axs.flatten()[i].axhline(y=RTOL, color='black', linestyle='dashed', linewidth=0.7)

    axs.flatten()[1].legend()

    print(f'jac_radii = {jac_radii}')
    print(f'max_eig_radii = {max_eig_radii}')

    

    plt.show()

    return

if __name__ == '__main__':
    main()
