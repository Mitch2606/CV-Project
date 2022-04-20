import numpy as np

A = np.array([1, 2, 3, 4, 5])


X = np.where(A == A.max())[0][0]

print(X)