# dosya adı: semi_analytical_ber.py
# Section IV (Analytical BER) figuru ve tablosu.
# Ciktilar: semi_analytical_ber.png , semi_analytical_ber.pdf
# Konsol: N, analitik BER, simulasyon BER, hata %

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import brentq

np.random.seed(42)

# ---- parameters (paper) ----
DT = 0.005
T_BIT = 0.2
K_ON = 50.0
K_OFF = 5.0
K_REC = 0.8 * 30.0 / (4.0**2)   # rule: 0.8*D/d^2 = 1.5
NUM_BITS = 5000
SAMPLE_TIME = 0.08
Q_MOLS = 50000
D_COEFF = 30.0
D_DIST = 4.0
POISSON_SCALE = 80.0
N_RECEPTORS_LIST = [50, 100, 200, 500, 1000, 2000, 5000]
NUM_ITER = 20


def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))


def build_fick_channel(bits):
    t = np.arange(0, len(bits) * T_BIT, DT)
    C = np.zeros_like(t)
    pt = np.maximum(np.arange(0, 2.0, DT), 1e-12)
    pulse = (Q_MOLS / ((4*np.pi*D_COEFF*pt)**1.5)) * np.exp(-(D_DIST**2)/(4*D_COEFF*pt))
    pulse *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT); e = min(s+len(pulse), len(C))
            C[s:e] += pulse[:e-s]
    return C


def solve_tm_ode(C):
    R = np.zeros_like(C); D = np.zeros_like(C); F = np.ones_like(C)
    for i in range(1, len(C)):
        R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1] - K_OFF*R[i-1])*DT, 0, 1)
        D[i] = np.clip(D[i-1] + (K_OFF*R[i-1] - K_REC*D[i-1])*DT, 0, 1)
        tot = R[i] + D[i]
        if tot > 1:
            R[i] /= tot; D[i] /= tot
        F[i] = 1 - R[i] - D[i]
    return R


def sample_signal(R):
    idx = [min(int((j*T_BIT+SAMPLE_TIME)/DT), len(R)-1) for j in range(NUM_BITS)]
    return np.array([R[i] for i in idx])


# ---- Step 1: signal means AND the Poisson-induced spread of mu ----
bits = np.random.randint(0, 2, NUM_BITS)
C_clean = build_fick_channel(bits)

# Clean (noise-free) means
R_clean = solve_tm_ode(C_clean)
s_clean = sample_signal(R_clean)
mu_1 = np.mean(s_clean[bits == 1])
mu_0 = np.mean(s_clean[bits == 0])

# Poisson-induced variance of the sampled Rb (independent of N):
# estimate it by passing several Poisson channel realizations through the ODE.
poiss1, poiss0 = [], []
for _ in range(NUM_ITER):
    Cn = np.random.poisson(C_clean * POISSON_SCALE) / POISSON_SCALE
    s = sample_signal(solve_tm_ode(Cn))
    poiss1.append(s[bits == 1])
    poiss0.append(s[bits == 0])
var_poiss1 = np.var(np.concatenate(poiss1))
var_poiss0 = np.var(np.concatenate(poiss0))

print(f"mu1 = {mu_1:.4f}, mu0 = {mu_0:.4f}")
print(f"Poisson var: bit1 = {var_poiss1:.2e}, bit0 = {var_poiss0:.2e}\n")


def optimal_threshold(mu1, mu0, s1, s0):
    def f(th):
        return (th-mu0)/(s0**2) - (mu1-th)/(s1**2)
    return brentq(f, mu0, mu1)


analytical, simulated = [], []
print(" N   | Analyt. BER | Sim BER  | Error %")
print("-" * 44)
for N in N_RECEPTORS_LIST:
    # total variance = receptor (binomial, ~1/N) + Poisson floor (N-independent)
    sigma1 = np.sqrt(mu_1*(1-mu_1)/N + var_poiss1)
    sigma0 = np.sqrt(mu_0*(1-mu_0)/N + var_poiss0)

    theta = optimal_threshold(mu_1, mu_0, sigma1, sigma0)
    ber_a = 0.5*q_func((mu_1-theta)/sigma1) + 0.5*q_func((theta-mu_0)/sigma0)
    analytical.append(ber_a)

    # Monte Carlo: both Poisson channel noise and receptor binomial noise
    ber_mc = 0.0
    for _ in range(NUM_ITER):
        Cn = np.random.poisson(C_clean*POISSON_SCALE)/POISSON_SCALE
        R_u = solve_tm_ode(Cn)
        R_n = np.random.binomial(N, R_u)/N
        s = sample_signal(R_n)
        ber_mc += np.mean((s > theta).astype(int) != bits)
    ber_mc /= NUM_ITER
    simulated.append(ber_mc)

    err = abs(ber_mc - ber_a)/max(ber_mc, 1e-9)*100
    print(f"{N:<4} | {ber_a:.5f}    | {ber_mc:.5f} | {err:.2f}")


plt.figure(figsize=(8, 6))
plt.plot(N_RECEPTORS_LIST, analytical, 'k-', lw=2.3, label="Semi-Analytical")
plt.plot(N_RECEPTORS_LIST, simulated, 'ro', ms=8, label="Monte Carlo")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of Receptors (N)")
plt.ylabel("BER")
plt.title("Semi-Analytical BER Validation")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("semi_analytical_ber.png", dpi=200)
plt.savefig("semi_analytical_ber.pdf")
plt.show()
