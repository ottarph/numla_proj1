import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from time import time
import matplotlib.pyplot as plt

from sparseitermethods import *
from matrix_builders import *
from polyak_itermethods import *

def jacobi_spectral_radius(A):
    '''
        Finds the spectral radius of G for the Jacobi method.
    '''

    A_1 = np.diag(np.diag(A))
    A_2 = A_1 - A

    A_1_inv = np.linalg.inv(A_1)

    G = A_1_inv @ A_2
    
    ss = np.linalg.eigvals(G)

    return np.max(np.abs(ss))


def f_gauss_seidel_spectral_radius(A):
    '''
        Finds the spectral radius of G for the forward Gauss-Seidel method.
    '''

    A_1 = np.tril(A)
    A_2 = A_1 - A

    A_1_inv = np.linalg.inv(A_1)

    G = A_1_inv @ A_2
    
    ss = np.linalg.eigvals(G)

    return np.max(np.abs(ss))


def successive_over_relaxation_spectral_radius(A, w):
    '''
        Finds the spectral radius of G for the successive over relaxation method.
    '''

    A_1 = np.diag(np.diag(A)) + w * np.tril(A, k=-1)
    A_2 = A_1 - A

    A_1_inv = np.linalg.inv(A_1)

    G = A_1_inv @ A_2
    
    ss = np.linalg.eigvals(G)

    return np.max(np.abs(ss))



def main():

    n = 10

    L = build_L_sparse(n).todense()

    print(jacobi_spectral_radius(L))
    
    print(f_gauss_seidel_spectral_radius(L))

    '''
    w = 1.5
    for w in np.linspace(1.561, 1.562, 11):
        print(w, successive_over_relaxation_spectral_radius(L, w))

    # w = 1.561 er good
    '''

    # Strikwerda 13.4, p. 354
    w = 2 / (1 + np.sin(np.pi / (n+1)) )
    print(successive_over_relaxation_spectral_radius(L, w))

    return


if __name__ == '__main__':
    main()
