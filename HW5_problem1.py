import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

np.set_printoptions(precision=6, suppress=True)

# b) a = b = 1
p = np.array([[0,   1,   0,   0,   0  ],
              [1/4, 0,   3/4, 0,   0  ],
              [0,   1/2, 0,   1/2, 0  ],
              [0,   0,   3/4, 0,   1/4],
              [0,   0,   0,   1,   0  ]])

q = np.zeros((61, 5))
q[0, 0] = 1
for n in range(60):
    q[n + 1] = q[n] @ p
print(q[50])
print(q[51])

avg = np.cumsum(q[:, 2]) / np.arange(1, 62)

# c) a = 0.3, b = 0.1
p2 = np.array([[0.7,   0.3,  0,     0,     0    ],
               [0.025, 0.75, 0.225, 0,     0    ],
               [0,     0.05, 0.8,   0.15,  0    ],
               [0,     0,    0.075, 0.85,  0.075],
               [0,     0,    0,     0.1,   0.9  ]])

A = np.vstack([p2.T - np.eye(5), np.ones(5)])
b = np.array([0, 0, 0, 0, 0, 1])
pi = np.linalg.lstsq(A, b, rcond=None)[0]
print(pi)
print(binom.pmf(range(5), 4, 0.75))

q2 = np.zeros((201, 5))
q2[0, 0] = 1
for n in range(200):
    q2[n + 1] = q2[n] @ p2
gap = np.abs(q2 - pi).max(axis=1)
print(np.argmax(gap < 1e-6))
print(np.sort(np.abs(np.linalg.eigvals(p2))))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(q[:, 2], label="$q_n(2)$")
ax[0].plot(q[:, 4], label="$q_n(4)$")
ax[0].plot(avg, "k", label="running average of $q_n(2)$")
ax[0].axhline(3/8, color="gray", linestyle="--", label=r"$\pi(2) = 3/8$")
ax[0].set_title("a = b = 1")
for k in range(5):
    ax[1].plot(q2[:, k], label=f"$q_n({k})$")
    ax[1].axhline(pi[k], color="gray", linestyle="--")
ax[1].set_title("a = 0.3, b = 0.1")
for a in ax:
    a.set_xlabel("n")
ax[0].set_ylim(0, 1)
ax[0].legend(loc="upper right", ncol=2)
ax[1].legend()
ax[0].set_ylabel("probability")
fig.tight_layout()
fig.savefig("hw5_p1.png", dpi=150)