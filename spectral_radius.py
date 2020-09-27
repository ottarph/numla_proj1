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

def maximum_eigenvalue_spectral_radius(A, Laplace=False):
    '''
        Finds the spectral radius of G for the successive over relaxation method.
    '''

    if Laplace:
        n = int(np.sqrt(x_0.shape[0]))

        rr = np.arange(1, n+1)
        v1 = np.sin(np.pi / (n + 1) * rr)

        V = np.outer(v1, v1)

        v = V.flatten()
        v = v / np.linalg.norm(v)

        A_d = np.diag(A.diagonal())

        u = (A - A_d) @ v

        s = 4 * np.cos(np.pi / (n+1))

        A_1 = A_d + s * np.outer(v,v)
        #A_1 = sp.sparse.csc_matrix(A_1)
        A_2 = A_1 - A

        A_1_inv = -0.25 * ( np.eye(n**2) + s / (4 - s) * np.outer(v,v) )
        #A_1_inv = sp.sparse.csc_matrix(A_1_inv)

    else:
        
        A_d = np.diag(np.diag(A))
        Amd = A - A_d

        S, V = np.linalg.eig(Amd)
        

        ind = np.argmax(S)

        s, v = S[ind], V[:,ind]

        A_1 = A_d + s * np.outer(v,v)

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
