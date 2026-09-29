import numpy as np
import matplotlib.pyplot as plt

P = np.array([[0,   1/2, 1/2, 0,   0  ],
              [1/4, 0,   0,   1/2, 1/4],
              [3/4, 0,   0,   0,   1/4],
              [0,   0,   0,   1,   0  ],
              [0,   0,   0,   0,   1  ]])
CUM = np.cumsum(P, axis=1) 
F, A = 3, 4

# exact answers from 1c and 4b
h    = [1/2, 5/8, 3/8]
g    = [4,   2,   4  ]
tauF = [4,   9/5, 5  ]
tauA = [4,   7/3, 17/5]

rng = np.random.default_rng(4560)
runs = 10_000
results = {}

print("start  h_hat   h      g_hat   g      tauF_hat tauF   tauA_hat tauA")
for x0, name in enumerate("UIM"):
    T, fold = [], []
    for _ in range(runs):
        x, t = x0, 0
        while x not in (F, A):
            u = rng.random()
            x = np.argmax(u < CUM[x])
            t += 1
        T.append(t)
        fold.append(x == F)
    T, fold = np.array(T), np.array(fold)
    results[name] = (T, fold)
    print(f"{name}      {fold.mean():.4f}  {h[x0]:.4f} {T.mean():.4f}  {g[x0]:.4f} "
          f"{T[fold].mean():.4f}   {tauF[x0]:.4f} {T[~fold].mean():.4f}   {tauA[x0]:.4f}")

Q, R = P[:3, :3], P[:3, 3:]
n = np.arange(1, 26)
pmf = np.array([(np.linalg.matrix_power(Q, k - 1) @ R)[1] for k in n]) 
pmf_F = pmf[:, 0] / h[1]
pmf_A = pmf[:, 1] / (1 - h[1])

T, fold = results["I"]
bins = np.arange(0.5, 26.5, 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
ax[0].hist(T[fold], bins=bins, density=True, alpha=0.6, label="simulated")
ax[0].plot(n, pmf_F, "ko", ms=4, label="exact")
ax[0].set_title("Start I, folded")
ax[1].hist(T[~fold], bins=bins, density=True, alpha=0.6, label="simulated")
ax[1].plot(n, pmf_A, "ko", ms=4, label="exact")
ax[1].set_title("Start I, aggregated")
for a in ax:
    a.set_xlabel("T (steps to absorption)")
    a.legend()
ax[0].set_ylabel("probability")
fig.tight_layout()
fig.savefig("hw4_p4.png", dpi=150)