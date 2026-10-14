import numpy as np
import matplotlib.pyplot as plt

def h1(R0, N, K):
    k = np.arange(1, K)
    rho = np.concatenate(([1], np.cumprod(N / (R0 * (N - k)))))
    return 1 / rho.sum()

# c)
R0s = np.linspace(0.5, 3, 200)
for N in (100, 1000):
    plt.plot(R0s, [h1(R0, N, N // 10) for R0 in R0s], label=f"exact $h_1$, N = {N}, K = {N // 10}")
plt.plot(R0s, np.maximum(0, 1 - 1/R0s), "k--", label=r"$\max(0, 1 - 1/R_0)$")
plt.xlabel("$R_0$")
plt.ylabel("P(outbreak reaches K | one case)")
plt.legend()
plt.tight_layout()
plt.savefig("hw6_p3.png", dpi=150)

# d)
N, K, gamma, dt = 100, 10, 1, 0.005
n = np.arange(1, K)
for R0 in (0.5, 0.8, 1.5, 3):
    beta = R0 * gamma / N
    p = beta * n * (N - n) * dt
    q = gamma * n * dt
    A = np.diag(p + q) - np.diag(p[:-1], 1) - np.diag(q[1:], -1)
    e = np.linalg.solve(A, np.ones(K - 1))
    print("R0 =", R0, " e1*dt =", round(e[0] * dt, 4), " h1 =", round(h1(R0, N, K), 4))

# simulation check, R0 = 1.5
rng = np.random.default_rng(4560)
beta = 1.5 * gamma / N
runs = 10_000
x = np.ones(runs, dtype=int)
T = np.zeros(runs)
alive = np.ones(runs, dtype=bool)
while alive.any():
    u = rng.random(runs)
    up = beta * x * (N - x) * dt
    down = gamma * x * dt
    x = x + alive * ((u < up).astype(int) - ((u >= up) & (u < up + down)).astype(int))
    T += alive
    alive &= (x > 0) & (x < K)
print("simulated e1*dt at R0 = 1.5:", round(T.mean() * dt, 4), "+-", round(T.std() * dt / np.sqrt(runs), 4))
