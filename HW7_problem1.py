import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

x = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97, 0.79, 1.50,
              2.00, 2.63, 1.19, -2.31, 0.87])
n = len(x)

def logg(a, b):
    return n * b - np.sum(np.log(np.exp(2*b) + (x - a)**2))

def iat(y):
    # 1 + 2 * (sum of autocorrelations), stop at the first negative one
    y = y - y.mean()
    tau, k = 1, 1
    while True:
        r = np.dot(y[:-k], y[k:]) / np.dot(y, y)
        if r < 0:
            return tau
        tau += 2 * r
        k += 1

# c)
rng = np.random.default_rng(4560)
T, burn = 50_000, 5000
chains = {}
for s in (0.02, 0.5, 5):
    z = np.zeros((T + 1, 2))
    acc = 0
    for t in range(T):
        y = z[t] + s * rng.standard_normal(2)
        if np.log(rng.random()) < logg(*y) - logg(*z[t]):
            z[t + 1] = y
            acc += 1
        else:
            z[t + 1] = z[t]
    chains[s] = z
    print("s =", s, "acceptance", acc / T, "IAT", iat(z[burn:, 0]))

fig, ax = plt.subplots(3, 1, sharex=True)
for k, s in enumerate(chains):
    ax[k].plot(chains[s][:400, 0])
    ax[k].set_ylabel(f"alpha, s={s}")
ax[2].set_xlabel("step")
fig.tight_layout()
fig.savefig("hw7_p1_trace.png", dpi=150)

a, b = chains[0.5][burn:, 0], chains[0.5][burn:, 1]
A, B = np.meshgrid(np.linspace(-2, 5, 400), np.linspace(-2.5, 2, 400))
G = np.exp(np.vectorize(logg)(A, B))
G /= G.sum()
pa, pb = G.sum(axis=0), G.sum(axis=1)
print("alpha chain:", a.mean(), np.percentile(a, [5, 95]))
print("alpha grid: ", (A[0] * pa).sum(), np.interp([0.05, 0.95], np.cumsum(pa), A[0]))
print("beta chain: ", np.exp(b).mean(), np.percentile(np.exp(b), [5, 95]))
print("beta grid:  ", (np.exp(B[:, 0]) * pb).sum(), np.exp(np.interp([0.05, 0.95], np.cumsum(pb), B[:, 0])))
print("sample mean", x.mean(), "sample median", np.median(x))

plt.figure()
plt.plot(a, b, ".", ms=1, alpha=0.2)
plt.contour(A, B, G, levels=8, colors="k")
plt.xlim(0, 3)
plt.ylim(-2, 1.5)
plt.xlabel("alpha")
plt.ylabel("log beta")
plt.savefig("hw7_p1_post.png", dpi=150)

# d)
rng = np.random.default_rng(4560)
iqr = lambda e: (np.percentile(e, 75) - np.percentile(e, 25)) / 1.349
for m in (10, 100, 1000):
    data = 1.5 + 0.8 * np.tan(np.pi * (rng.random((2000, m)) - 0.5))
    print("n =", m, "mean", iqr(data.mean(axis=1)), "median", iqr(np.median(data, axis=1)))
