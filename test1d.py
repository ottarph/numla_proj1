import numpy as np
import numpy.linalg as npl
import scipy as sp
import matplotlib.pyplot as plt

n = 9
h = 1 / (n+1)

print(h)

f = lambda x: 0*x + 1
u_ex_1d = lambda x: 0.5*x**2 - 0.5*x

xx = np.linspace(0, 1, n+2)[1:-1]

print(xx)

print(f(xx))

A = np.diag(np.full(n, 2, dtype=float))
A += np.diag(np.full(n-1, -1, dtype=float), -1)
A += np.diag(np.full(n-1, -1, dtype=float), +1)
A = -A
print(A)

u = np.empty(n+2, dtype=float)
u[1:-1] = npl.solve(A, h**2 * f(xx))
u[0], u[-1] = [0, 0]
x = np.empty(n+2, dtype=float)
x[1:-1] = xx
x[0], x[-1] = [0, 1]

plt.plot(x, u)
plt.plot(x, u_ex_1d(x))
plt.show()






