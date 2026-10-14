import csv
import numpy as np
import matplotlib.pyplot as plt

def h(x):
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(abs(2*x - 1))

games = list(csv.DictReader(open("big12.csv")))
teams = sorted({g["home"] for g in games} | {g["away"] for g in games})
m = len(teams)
idx = {t: i for i, t in enumerate(teams)}

def results(season, before_week=99):
    out = []
    for g in games:
        if g["season"] == season and g["completed"] == "1" and int(g["week"]) < before_week:
            i, j = idx[g["home"]], idx[g["away"]]
            S, T = int(g["home_pts"]), int(g["away_pts"])
            out.append((i, j, h((S + 1) / (S + T + 2))))
            out.append((j, i, h((T + 1) / (S + T + 2))))
    return out

def keener(Pi, w=1):
    A = w * Pi.copy()
    n = np.zeros(m)
    for i, j, hij in results("2026", before_week=6):   # games before Oct 10
        A[i, j] += hij
        n[i] += 1
    np.fill_diagonal(A, 0)
    A = A / (n + w * (m - 1))[:, None]
    lam, V = np.linalg.eig(A)
    k = np.argmax(lam.real)
    r = np.abs(V[:, k].real)
    lam_abs = np.sort(np.abs(lam))[::-1]
    return r / r.sum(), lam[k].real, lam_abs[1] / lam_abs[0]

# priors
flat = np.full((m, m), 0.5)

tot = np.zeros((m, m))
cnt = np.zeros((m, m))
for i, j, hij in results("2025"):
    tot[i, j] += hij
    cnt[i, j] += 1
last = np.full((m, m), 0.5)
last[cnt > 0] = tot[cnt > 0] / cnt[cnt > 0]

wins = np.zeros(m)
played = np.zeros(m)
for i, j, hij in results("2025"):
    wins[i] += hij > 0.5
    played[i] += 1
s = (wins + 1) / (played + 2)          # 2025 conference win fraction, +1/+2 so nobody is 0
own = s[:, None] / (s[:, None] + s[None, :])

priors = {"flat": flat, "last season": last, "2025 record (BT)": own}
R = {}
for name, Pi in priors.items():
    r, lam, ratio = keener(Pi)
    R[name] = r
    print(f"{name}: lambda* = {lam:.4f}, |lambda2/lambda*| = {ratio:.4f}")
    print("  ", [teams[i] for i in np.argsort(-r)])

# predict Oct 10-11 (week 6), big12.csv was re-fetched after the weekend
week6 = [g for g in games if g["season"] == "2026" and g["week"] == "6"]
for name, r in R.items():
    correct = 0
    for g in week6:
        i, j = idx[g["home"]], idx[g["away"]]
        pick = g["home"] if r[i] > r[j] else g["away"]
        winner = g["home"] if int(g["home_pts"]) > int(g["away_pts"]) else g["away"]
        correct += pick == winner
        print(f"  {name}: {g['away']} at {g['home']}  pick {pick}, actual {winner} "
              f"({g['away_pts']}-{g['home_pts']})")
    print(name, "correct:", correct, "/", len(week6))

x = np.arange(m)
plt.figure(figsize=(12, 4.5))
for k, (name, r) in enumerate(R.items()):
    plt.bar(x + (k - 1) * 0.27, r, width=0.27, label=name)
plt.xticks(x, teams, rotation=60, ha="right")
plt.ylabel("r (sums to 1)")
plt.title("Keener ranking, 2026 Big 12 games before Oct 10, w = 1")
plt.ylim(0, 0.1)
plt.legend(loc="upper left", ncol=3)
plt.tight_layout()
plt.savefig("hw6_p2.png", dpi=150)
