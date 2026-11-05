import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

lam = lambda t, a=12, b=8, phi=16: a + b * np.cos(np.pi * (t - phi) / 12)
Lam = lambda s, t, a=12, b=8, phi=16: a * (t - s) + 12 * b / np.pi * (np.sin(np.pi * (t - phi) / 12) - np.sin(np.pi * (s - phi) / 12))

T = 720
rng = np.random.default_rng(4560)
prop = np.cumsum(rng.exponential(1 / 20, int(20 * T * 1.1)))   # rate 20 proposals
prop = prop[prop <= T]
t = prop[rng.random(len(prop)) < lam(prop) / 20]                # keep with prob lam(t)/20
n = len(t)
print("proposals", len(prop), "kept", n, "fraction", n / len(prop))

hour = np.floor(t % 24).astype(int)
day = np.floor(t / 24).astype(int)
n34 = np.bincount(day[hour == 3], minlength=30)
print("3-4am: mean", n34.mean(), "var", n34.var(ddof=1), "Lambda(3,4)", Lam(3, 4))

def nll(p):
    a, b, phi = p
    if b < 0 or b >= a:
        return np.inf
    return Lam(0, T, a, b, phi) - np.sum(np.log(lam(t, a, b, phi)))
fit = minimize(nll, (10, 5, 14), method="Nelder-Mead")
print("fit (a, b, phi)", fit.x, "log-lik", -fit.fun)
print("constant n/T", n / T, "log-lik", n * np.log(n / T) - n)

plt.bar(np.arange(24) + 0.5, np.bincount(hour, minlength=24) / 30, width=1, alpha=0.5, label="mean arrivals per clock hour")
tt = np.linspace(0, 24, 200)
plt.plot(tt, lam(tt), "k", label="lambda(t)")
plt.xlabel("hour of day")
plt.legend()
plt.savefig("hw8_p3.png", dpi=150)
