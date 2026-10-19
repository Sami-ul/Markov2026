import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma, binom

lam, H = 4, 200_000
rng = np.random.default_rng(4560)

T = np.cumsum(rng.exponential(1 / lam, size=(H, 40)), axis=1)# exponential gaps
N1 = (T <= 1).sum(axis=1)
N2 = rng.poisson(lam, H)# Poisson count + uniform points
print("gaps:    mean", N1.mean(), "var", N1.var())
print("uniform: mean", N2.mean(), "var", N2.var())

Nhalf = (T <= 0.5).sum(axis=1)
c = N1 == 3
print("P(N(1/2)=0 | N(1)=3):", (Nhalf[c] == 0).mean())
print("E[T1 | N(1)=3]:", T[c, 0].mean())
print("corr of half hour counts:", np.corrcoef(Nhalf, N1 - Nhalf)[0, 1])

dts = 1 / np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
k = np.arange(100)
tv = [0.5 * np.abs(binom.pmf(k, round(1 / dt), lam * dt) - poisson.pmf(k, lam)).sum() for dt in dts]
print("TV slope:", np.polyfit(np.log10(dts), np.log10(tv), 1)[0])

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].eventplot([T[i][T[i] <= 1] for i in range(10)] + [np.sort(rng.random(n)) for n in N2[:10]])
ax[0, 0].set_xlabel("t (hours); rows 0-9 exp gaps, rows 10-19 count + uniform")
ax[0, 1].hist(N1, bins=np.arange(-0.5, 16), density=True, alpha=0.5)
ax[0, 1].hist(N2, bins=np.arange(-0.5, 16), density=True, alpha=0.5)
ax[0, 1].plot(range(16), poisson.pmf(range(16), lam), "ko")
ax[0, 1].set_xlabel("N(1)")
t = np.linspace(0, 3, 300)
ax[1, 0].hist(T[:, 2], bins=100, density=True, alpha=0.6)
ax[1, 0].plot(t, gamma.pdf(t, 3, scale=1 / lam), "k")
ax[1, 0].set_xlabel("T3 (hours)")
ax[1, 1].loglog(dts, tv, "o")
ax[1, 1].set_xlabel("dt (hours)")
ax[1, 1].set_ylabel("TV distance")
fig.tight_layout()
fig.savefig("hw7_p3.png", dpi=150)
