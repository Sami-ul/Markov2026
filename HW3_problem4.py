import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def survival(N, R, T, d0, seed):
    rng = np.random.default_rng(seed)
    lamb  = np.zeros(R, dtype=int)
    lions = np.full((N, R), d0, dtype=int)
    alive = np.ones(R, dtype=bool)
    S = np.empty(T + 1)
    S[0] = 1.0
    for t in range(1, T + 1):
        random_lambs = (2*rng.integers(0, 2, size=R))-1
        random_lions = (2*rng.integers(0, 2, size=(N, R)))-1
        lamb += random_lambs
        lions += random_lions
        caught = (lions == lamb).any(axis=0)
        alive &= ~caught
        S[t] = alive.mean()
    return S

R, T, d0 = 20_000, 10_000, 10
S1 = survival(1, R, T, d0, seed=1)
S2 = survival(2, R, T, d0, seed=2)

t = np.arange(1, T + 1)
m = (t >= 100) & (t <= 10_000)
print(S1[[100, 1000, 10000]])
def fit_beta(S):
    x = np.log10(t[m])
    y = np.log10(S[1:][m])
    slope = np.polyfit(x, y, 1)[0]
    return -slope

b1, b2 = fit_beta(S1), fit_beta(S2)
print(b1, b2)

plt.figure(figsize=(7, 5))
plt.loglog(t, S1[1:], label=r"$S_1(t)$, one lion")
plt.loglog(t, erf(d0 / (2*np.sqrt(t))), "--", label=r"erf$(d_0/2\sqrt{t})$")
plt.loglog(t, S2[1:], label=r"$S_2(t)$, two lions")
plt.loglog(t, S1[1:]**2, ":", label=r"$S_1(t)^2$")
plt.xlabel("t (steps)")
plt.ylabel("survival probability")
plt.title(rf"$\beta_1$ = {b1:.3f}, $\beta_2$ = {b2:.3f}, fit over $10^2 \leq t \leq 10^4$; R={R}, $d_0$={d0}")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("hw3_p4.png", dpi=150, bbox_inches="tight")

for tt in (100, 1000, 10000):
    print(tt, S2[tt], S1[tt]**2)