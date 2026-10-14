import numpy as np

def wielandt(m):
    p = np.zeros((m, m))
    for i in range(m - 1):
        p[i, i + 1] = 1
    p[m - 1, 0] = 1/2
    p[m - 1, 1] = 1/2
    return p

print("m   M   (m-1)^2+1   |lambda2|")
for m in range(3, 13):
    p = wielandt(m)
    B = p > 0
    M = 1
    while not B.all():
        B = (B @ p) > 0
        M += 1
    lam = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    print(m, M, (m - 1)**2 + 1, round(lam[1], 4))
