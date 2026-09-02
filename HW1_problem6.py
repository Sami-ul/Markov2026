import random
import math
import os
import matplotlib.pyplot as plt

analytic = 23 * math.pi / 192
random.seed(4560)

N_values = [
    100,
    300,
    1000,
    3000,
    10000,
    30000,
    100000,
    300000,
    1000000
]

estimates = []

for N in N_values:
    count = 0

    for _ in range(N):
        x = random.random()
        y = random.random()
        z = random.random()

        if x**2 + y**2 < z and z**2 > x*y:
            count += 1

    estimate = count / N
    estimates.append(estimate)

    print("N =", N, "estimate =", estimate)

print("analytic =", analytic)
print("final absolute error =", abs(estimates[-1] - analytic))

plt.plot(N_values, estimates, marker="o")

plt.axhline(
    analytic,
    linestyle="--",
    label=f"Analytic = {analytic:.4f}"
)

plt.xscale("log")
plt.xlabel("Sample size N")
plt.ylabel("Estimated probability")
plt.title("Monte Carlo Estimate vs Sample Size")
plt.legend()
plt.tight_layout()
os.makedirs("hw1", exist_ok=True)
plt.savefig("hw1/HW1_problem6.png", dpi=300, bbox_inches="tight")
plt.show()
