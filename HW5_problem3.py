import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=120)

p = np.array([[0,   1/2, 1/2, 0,   0,   0,   0,   0],
              [0,   0,   1/2, 1/2, 0,   0,   0,   0],
              [1/2, 0,   0,   0,   1/2, 0,   0,   0],
              [1/3, 0,   1/3, 0,   1/3, 0,   0,   0],
              [0,   1/3, 0,   0,   0,   1/3, 1/3, 0],
              [0,   0,   0,   0,   0,   1,   0,   0],
              [0,   0,   0,   0,   0,   0,   0,   1],
              [0,   0,   0,   0,   0,   0,   1,   0]])

# a)
q0 = np.ones(8) / 8
print(q0 @ np.linalg.matrix_power(p, 100))
print(q0 @ np.linalg.matrix_power(p, 101))

# b)
d = 0.85
G = d * p + (1 - d) / 8 * np.ones((8, 8))

q = q0
n = 0
while True:
    q_next = q @ G
    n += 1
    if np.abs(q_next - q).sum() < 1e-10:
        break
    q = q_next
pi = q_next
print(n)
print(pi)

A = np.vstack([G.T - np.eye(8), np.ones(8)])
b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
print(np.linalg.lstsq(A, b, rcond=None)[0])
print(np.argsort(-pi) + 1)

# c)
rng = np.random.default_rng(4560)
R, T = 100, 10**5
CUM = np.cumsum(G, axis=1)
x = np.zeros(R, dtype=int)
counts = np.zeros((R, 8))
Ts = np.unique(np.logspace(1, 5, 41).astype(int))
err = []
for t in range(1, T + 1):
    u = rng.random(R)
    x = np.argmax(u[:, None] < CUM[x], axis=1)
    counts[np.arange(R), x] += 1
    if t in Ts:
        pi_hat = counts / t
        err.append(np.sqrt(np.mean(np.abs(pi_hat - pi).max(axis=1) ** 2)))
err = np.array(err)

fit = Ts >= 100
slope, intercept = np.polyfit(np.log10(Ts[fit]), np.log10(err[fit]), 1)
iid = np.sqrt(pi[5] * (1 - pi[5]) / T)
print(counts[0] / T)
print(slope)
print(err[-1], iid, err[-1] / iid)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pages = np.arange(1, 9)
ax[0].bar(pages - 0.2, counts[0] / T, width=0.4, label="one surfer")
ax[0].bar(pages + 0.2, pi, width=0.4, label=r"$\pi$")
ax[0].set_xticks(pages)
ax[0].set_xlabel("page")
ax[0].set_ylabel("probability")
ax[0].set_title("One surfer vs $\\pi$, $T = 10^5$")
ax[0].legend()
ax[1].loglog(Ts, err, "o", ms=4, label="rms over surfers")
ax[1].loglog(Ts[fit], 10**intercept * Ts[fit]**slope, "k-", label=f"fit, slope = {slope:.3f}")
ax[1].set_xlabel("T (steps)")
ax[1].set_ylabel(r"rms of $\max_i |\hat\pi_i - \pi_i|$")
ax[1].set_title("Error vs T")
ax[1].legend()
fig.tight_layout()
fig.savefig("hw5_p3.png", dpi=150)