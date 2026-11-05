import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

H, w, q = 200_000, 5, 0.4
rng = np.random.default_rng(4560)

star1 = rng.binomial(rng.poisson(3 * w, H), q)   # emit every photon, keep each with prob q
sky1 = rng.binomial(rng.poisson(2 * w, H), q)
star2, sky2 = rng.poisson(1.2 * w, H), rng.poisson(0.8 * w, H)   # detected processes directly

for name, s, k in (("thinned", star1, sky1), ("direct", star2, sky2)):
    tot = s + k
    print(name, "star", s.mean(), s.var(), "sky", k.mean(), k.var(), "corr", np.corrcoef(s, k)[0, 1])
    print("   total", tot.mean(), tot.var(), "P(6 star | 10)", (s[tot == 10] == 6).mean())

# first detection of each kind: thin the photon times, or draw Exp(1.2), Exp(0.8) directly
tstar = np.cumsum(rng.exponential(1 / 3, size=(H, 40)), axis=1)
tsky = np.cumsum(rng.exponential(1 / 2, size=(H, 40)), axis=1)
S1 = tstar[np.arange(H), (rng.random((H, 40)) < q).argmax(axis=1)]
K1 = tsky[np.arange(H), (rng.random((H, 40)) < q).argmax(axis=1)]
S2, K2 = rng.exponential(1 / 1.2, H), rng.exponential(1 / 0.8, H)
for name, S, K in (("thinned", S1, K1), ("direct", S2, K2)):
    first, star = np.minimum(S, K), S < K
    print(name, "P(star first)", star.mean(), "within 1 s", (star & (first <= 1)).mean(),
          "E[M]", np.maximum(S, K).mean(), "corr(time, type)", np.corrcoef(first, star)[0, 1])

k = np.arange(25)
plt.hist(star1 + sky1, bins=k - 0.5, density=True, alpha=0.5, label="thinned")
plt.hist(star2 + sky2, bins=k - 0.5, density=True, alpha=0.5, label="direct")
plt.plot(k, poisson.pmf(k, 10), "ko", label="Poisson(10)")
plt.xlabel("detections in 5 s")
plt.legend()
plt.savefig("hw8_p2.png", dpi=150)
