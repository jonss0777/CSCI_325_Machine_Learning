import numpy as np

A = np.array([[1,2,3], [5,5,5]])

B = np.array([[9,9,9], [2,1,2]])

M = A.shape[0]
N = A.shape[1]

print(A.T @ A)
k0 = A.T @ A
print(-2*(A.T@B))
k1 = -2*(A.T@B)
print(B.T @ B)
k2 = B.T @ B

r = k0-k1-k2
print(r)
print(np.square(r))
print(np.sqrt(r))
#0 - F1

