import numpy as np

p = np.array([[9/10, 1/10, 0],
              [0,    3/4,  1/4],
              [1/2,  0,    1/2]])

p50 = np.linalg.matrix_power(p, 50)
np.set_printoptions(precision=6, suppress=True)
print(p50)
