import numpy as np
import numpy.linalg as npl
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib

n = 40
h = 1 / (n+1)


f = lambda x, y: 0*x + 1
#f = lambda x, y: 10*x + 1

x_line = np.linspace(0, 1, n+2)[1:-1]
y_line = np.linspace(0, 1, n+2)[1:-1]

U = np.arange(1, n*n+1).reshape((n,n))

X, Y = np.meshgrid(x_line, y_line)

x, y, u = X.flatten(), Y.flatten(), U.flatten()

if n <= 15:

    for i in range(n*n):
        plt.text(x[i], y[i], u[i])

    plt.show()


def build_L(n):
    B = np.diag(np.full(n, -4, dtype=float))
    B += np.diag(np.full(n-1, 1, dtype=float), -1)
    B += np.diag(np.full(n-1, 1, dtype=float), +1)

    I = np.eye(n)

    L = np.zeros((n*n,n*n), dtype=float)

    L[:n,:n] = B
    for i in range(1, n):
        L[(i-1)*n:i*n, i*n:(i+1)*n] = I
        L[i*n:(i+1)*n, i*n:(i+1)*n] = B
        L[i*n:(i+1)*n, (i-1)*n:i*n] = I
    
    return L

L = build_L(n)
b = h**2 * f(x,y)

from sparseitermethods import *
L_s = build_L_sparse(n)
u_0 = np.ones_like(x)
u_0 = np.zeros_like(x)

w = 1.5
u, i, r = succesive_over_relaxation_sparse(L_s, b, u_0, w)
#u, i, r = f_gauss_seidel_sparse_fp(L_s, b, u_0)
#u, i, r = jacobi_fp_sparse(L_s, b, u_0)
print(i)
#u = npl.solve(-L, -b)

xx, yy = np.meshgrid(np.linspace(0,1,n+2), np.linspace(0,1,n+2))
uu = np.zeros((n+2,n+2))
uu[1:-1,1:-1] = u.reshape((n,n))

fig = plt.figure(figsize=(8, 6), dpi=100)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, uu, rstride=1, cstride=1, cmap=cm.viridis)                    

plt.show()

