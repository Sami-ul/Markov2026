import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

lam, mu, T = 12, 3, 500
rng = np.random.default_rng(4560)
t, n = 0.0, 0
time_in = np.zeros(40)
while t < T:
    rate = lam + n * mu
    dt = min(rng.exponential(1 / rate), T - t)   # cut the last holding time at T
    time_in[n] += dt
    t += dt
    if rng.random() < lam / rate:
        n += 1
    else:
        n -= 1
frac = time_in / T
k = np.arange(len(frac))
d = np.abs(frac - poisson.pmf(k, 4))
print("largest difference", d.max(), "at n =", d.argmax())
print("time average of X", (k * frac).sum(), " rho =", lam / mu)
print("P(X > c):", [(c, round(poisson.sf(c, 4), 4)) for c in range(6, 11)])
print("smallest c with P(X > c) <= 0.01:", next(c for c in range(50) if poisson.sf(c, 4) <= 0.01))

plt.bar(k[:16] - 0.2, frac[:16], width=0.4, label="fraction of time, simulation")
plt.bar(k[:16] + 0.2, poisson.pmf(k[:16], 4), width=0.4, label="Poisson(4)")
plt.xlabel("bikes out")
plt.ylabel("fraction of time")
plt.legend()
plt.savefig("hw10_p2.png", dpi=150)
