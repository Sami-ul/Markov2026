import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

Q = np.array([[-2, 2, 0, 0], [1, -3, 2, 0], [0, 2, -4, 2], [0, 0, 3, -3.]])
pi = np.array([3, 6, 6, 4]) / 19
print("Q^T pi", Q.T @ pi)

ev = np.sort(np.linalg.eigvals(Q).real)[::-1]
print("eigenvalues", ev, "sum", ev.sum(), "trace", np.trace(Q))
l2 = ev[1]

t = np.linspace(0, 5, 1001)
x = np.array([expm(s * Q)[0] for s in t])   # start empty: row 0 of e^{tQ}
TV = 0.5 * np.abs(x - pi).sum(axis=1)
print("TV = 0.01 at t =", t[np.argmax(TV <= 0.01)], " ln(100)/|l2| =", np.log(100) / abs(l2))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i in range(4):
    ax[0].plot(t, x[:, i], label=f"x{i}(t)")
ax[0].set_xlabel("t (hours)")
ax[0].legend()
ax[1].semilogy(t, TV, label="TV(t)")
ax[1].semilogy(t, TV[0] * np.exp(l2 * t), "k--", label="slope lambda_2")
ax[1].set_xlabel("t (hours)")
ax[1].legend()
fig.savefig("hw9_p3.png", dpi=150)
