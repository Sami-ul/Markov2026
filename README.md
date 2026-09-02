# Markov2026

Runnable code for APPM 4560 (Markov Processes), labeled by homework and problem.
Assignment PDFs, LaTeX writeups, generated plots, and local environments are kept
out of the repository.

## HW1 Problem 6

`HW1_problem6.py` compares the analytic probability for the coupled inequalities
with a Monte Carlo estimate and creates the required convergence plot.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\HW1_problem6.py
```

The script prints the estimates and comparison in the terminal, then saves the
plot to `hw1/HW1_problem6.png`.
