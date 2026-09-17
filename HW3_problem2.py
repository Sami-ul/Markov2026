import numpy as np

p = np.array([[1/2, 1/2, 0, 0, 0, 0],
              [1/3, 0, 1/3, 0, 1/3, 0],
              [0, 0, 1/4, 3/4, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0]])

p20 = np.linalg.matrix_power(p, 20)
p21 = np.linalg.matrix_power(p, 21)
np.set_printoptions(precision=6, suppress=True)
print(p20)
print(p21)
