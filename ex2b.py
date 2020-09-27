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
        Finding approximate values for optimal h and lambda in Jacobi heavy ball method
    '''


    n = 10

    TOL = 1e-7
    RTOL = TOL

    L = build_L_sparse(n)

    dx = 1 / (n + 1)
    b = np.ones(n**2) * dx**2
    x_0 = np.ones(n**2)

    
    N = 5
    it_min = 1000
    l_min, h_min = 0, 0
    iterations = np.zeros((N,N))
    ll = np.linspace(0, 2, N)
    hh = np.linspace(0, 2, N)
    for i, l in enumerate(ll):
        print(f'{i} / {len(ll)}')
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
        print(f'{i} / {len(ll)}')
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
    
    i,j = np.unravel_index(np.argmin(iterations, axis=None), iterations.shape)

    print(f'l_min = {l_min}, h_min = {h_min}, it_min = {it_min}')

    lll, hhh = np.meshgrid(ll, hh)
    

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(lll, hhh, iterations, rstride=1, cstride=1, cmap=cm.Greys)   
    plt.xlabel("$l$")
    plt.ylabel("$h$")
    ax.set_zlabel('Iterations')

    plt.show()

    return

if __name__ == '__main__':
    main()
