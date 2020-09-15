import numpy as np
import scipy as sp
from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized
from timeit import default_timer as timer

def jacobi(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = np.diag(np.diag(A))
    A_2 = A_1 - A

    r_0 = np.linalg.norm(A @ x_0 - b)

    start = timer()
    S = np.linalg.inv(A_1)
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    x_k = np.copy(x_0)
    A_1x = A_2 @ x_k + b
    #x = np.linalg.solve(A_1, A_1x)
    x = S @ A_1x
    r = np.linalg.norm(A @ x - b)

    i = 1

    while r > tol and r / r_0 > rtol:
        
        i += 1

        x_k = x
        A_1x = A_2 @ x_k + b
        #x = np.linalg.solve(A_1, A_1x)
        x = S @ A_1x
        r = np.linalg.norm(A @ x - b)

    return x, i


def f_gauss_seidel(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = np.tril(A)
    A_2 = A_1 - A

    r_0 = np.linalg.norm(A @ x_0 - b)

    start = timer()
    S = np.linalg.inv(A_1)
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    x_k = np.copy(x_0)
    A_1x = A_2 @ x_k + b
    #x = np.linalg.solve(A_1, A_1x)
    x = S @ A_1x
    r = np.linalg.norm(A @ x - b)

    i = 1

    while r > tol and r / r_0 > rtol:
        
        i += 1

        x_k = x
        A_1x = A_2 @ x_k + b
        #x = np.linalg.solve(A_1, A_1x)
        x = S @ A_1x
        r = np.linalg.norm(A @ x - b)

    return x, i

def succesive_over_relaxation(A, b, x_0, w=0.5, tol=1e-7, rtol=1e-7):

    A_1 = w * np.tril(A) + (1 - w) * np.diag(np.diag(A))
    A_2 = A_1 - A

    r_0 = np.linalg.norm(A @ x_0 - b)

    start = timer()
    #S = np.linalg.inv(A_1)
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    x_k = np.copy(x_0)
    A_1x = A_2 @ x_k + b
    x = np.linalg.solve(A_1, A_1x)
    #x = S @ A_1x
    r = np.linalg.norm(A @ x - b)

    i = 1

    while r > tol and r / r_0 > rtol:
        
        i += 1

        x_k = x
        A_1x = A_2 @ x_k + b
        x = np.linalg.solve(A_1, A_1x)
        #x = S @ A_1x
        r = np.linalg.norm(A @ x - b)

    return x, i

def jacobi_fp(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = np.diag(np.diag(A))
    A_2 = A_1 - A

    start = timer()
    A_1_inv = np.linalg.inv(A_1)
    #A_1_inv = np.diag(1 / np.diag(A_1))
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    r_0 = np.linalg.norm(A @ x_0 - b)

    x_k = np.copy(x_0)
    x = G @ x_k + f
    r = np.linalg.norm(A @ x - b)

    i = 1

    while r > tol and r / r_0 > rtol:
        
        i += 1

        x_k = x
        x = G @ x_k + f
        r = np.linalg.norm(A @ x - b)

    return x, i

def jacobi_fp_sparse(A, b, x_0, tol=1e-7, rtol=1e-7):

    A_1 = sp.sparse.diags(A.diagonal())
    A_2 = A_1 - A

    start = timer()
    A_1_inv = sp.sparse.linalg.inv(A_1.tocsc()).todok()
    #A_1_inv = np.diag(1 / np.diag(A_1))
    end = timer()
    print(f'{(end-start)*1e3:.2f}ms invertion')

    G = A_1_inv @ A_2
    f = A_1_inv @ b

    r_0 = np.linalg.norm(A @ x_0 - b)

    x_k = np.copy(x_0)
    x = G @ x_k + f
    r = np.linalg.norm(A @ x - b)

    i = 1

    while r > tol and r / r_0 > rtol:
        
        i += 1

        x_k = x
        x = G @ x_k + f
        r = np.linalg.norm(A @ x - b)

    return x, i


def main():

    
    np.random.seed(0)

    n = 150

    A = np.diag(np.full(n, 2, dtype=float), 0)
    A += np.diag(np.full(n-1, -1, dtype=float), -1)
    A += np.diag(np.full(n-1, -1, dtype=float), +1)

    x = np.arange(1, n+1, dtype=float)
    b = A @ x

    #'''
    start = timer()
    x_jac, i_jac = jacobi(A, b, np.ones(n))
    end = timer()
    print('Jac', i_jac, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_jac))
    #'''

    '''
    start = timer()
    x_gs, i_gs = f_gauss_seidel(A, b, np.ones(n))
    end = timer()
    print('GS ', i_gs, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_gs))
    '''

    '''
    w = 1.2
    start = timer()
    x_sor, i_sor = succesive_over_relaxation(A, b, np.ones(n), w)
    end = timer()
    print('SOR', i_sor, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_sor))

    start = timer()
    x_ex = np.linalg.solve(A, b)
    end = timer()
    print('Ex ', '_'*int(np.log10(i_sor)+1), f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_ex))
    '''

    start = timer()
    x_jfp, i_jfp = jacobi_fp(A, b, np.ones(n))
    end = timer()
    print('Jfp', i_jfp, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_jfp))



    A_s = dok_matrix(A)

    start = timer()
    x_js, i_js = jacobi_fp_sparse(A_s, b, np.ones(n))
    end = timer()
    print('Js ', i_js, f'{(end - start)*1e0:.2f} s', np.linalg.norm(x - x_js))
    

    return

if __name__ == '__main__':
    main()
