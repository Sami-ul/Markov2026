import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from itertools import combinations

np.set_printoptions(precision=4, suppress=True)

@lru_cache(None)
def EM(S, lam):
    # expected time until everyone in group S is done
    tot = sum(lam[i] for i in S)
    return 1 / tot + sum(lam[i] / tot * EM(S - {i}, lam) for i in S if len(S) > 1)

@lru_cache(None)
def last(S, j, lam):
    # probability student j is last, starting from group S
    if len(S) == 1:
        return 1.0
    tot = sum(lam[i] for i in S)
    return sum(lam[i] / tot * last(S - {i}, j, lam) for i in S if i != j)

lam = (3, 2, 1.5, 1.2, 1, 0.75)
everyone = frozenset(range(6))
print("E[M] groups:", EM(everyone, lam))
print("E[M] alternating:", sum((-1)**(r + 1) * sum(1 / sum(g) for g in combinations(lam, r)) for r in range(1, 7)))
print("P(last):", np.array([last(everyone, j, lam) for j in range(6)]))

rng = np.random.default_rng(4560)
X = rng.exponential(1 / np.array(lam), size=(200_000, 6))
M = X.max(axis=1)
print("E[M] sim:", M.mean())
print("P(last) sim:", np.bincount(X.argmax(axis=1)) / len(M))
print("mean first finish time by who was first:",
      np.array([X.min(axis=1)[X.argmin(axis=1) == i].mean() for i in range(6)]), "1/Lambda =", 1 / sum(lam))

ks = range(1, 13)
EMk = [EM(frozenset(range(k)), tuple(range(1, k + 1))) for k in ks]
print("E[M] for rates 1..k:", np.array(EMk))

t = np.linspace(0, 8, 400)
dens = np.gradient(np.prod(1 - np.exp(-np.outer(t, lam)), axis=1), t)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(M, bins=100, density=True, alpha=0.6)
ax[0].plot(t, dens, "k")
ax[0].set_xlim(0, 8)
ax[0].set_xlabel("M (hours)")
ax[1].plot(ks, EMk, "o-")
ax[1].set_xlabel("k")
ax[1].set_ylabel("E[M]")
fig.savefig("hw7_p2.png", dpi=150)
