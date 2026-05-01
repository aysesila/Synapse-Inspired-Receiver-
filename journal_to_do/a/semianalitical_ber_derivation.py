import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import brentq


#  settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "lines.linewidth": 2.3,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.dpi": 140,
})

np.random.seed(42)

# parameters

DT = 0.005
T_BIT = 0.2

K_ON = 50.0
K_OFF = 5.0
K_REC = 0.45

NUM_BITS = 5000
SAMPLE_TIME = 0.08

Q_MOLS = 50000
D_COEFF = 30.0
D_DIST = 4.0

POISSON_SCALE = 80.0

N_RECEPTORS_LIST = [50, 100, 200, 500, 1000, 2000, 5000]
NUM_ITER = 20

# functions

def q_func(x):
    return 0.5 * erfc(x / np.sqrt(2))


def build_fick_channel(bits):
    total_time = len(bits) * T_BIT
    t = np.arange(0, total_time, DT)
    C = np.zeros_like(t)

    pulse_t = np.arange(0, 2.0, DT)
    t_safe = np.maximum(pulse_t, 1e-12)

    pulse = (Q_MOLS / ((4*np.pi*D_COEFF*t_safe)**1.5)) \
            * np.exp(-(D_DIST**2)/(4*D_COEFF*t_safe))
    pulse *= 0.005

    for i, b in enumerate(bits):
        if b == 1:
            start = int(i*T_BIT/DT)
            end = min(start+len(pulse), len(C))
            C[start:end] += pulse[:end-start]

    return C


def solve_tm_ode(C):
    R = np.zeros_like(C)
    D = np.zeros_like(C)
    F = np.ones_like(C)

    for i in range(1, len(C)):
        dR = (K_ON*C[i-1]*F[i-1] - K_OFF*R[i-1]) * DT
        dD = (K_OFF*R[i-1] - K_REC*D[i-1]) * DT

        R[i] = np.clip(R[i-1] + dR, 0, 1)
        D[i] = np.clip(D[i-1] + dD, 0, 1)

        total = R[i] + D[i]
        if total > 1:
            R[i] /= total
            D[i] /= total

        F[i] = 1 - R[i] - D[i]

    return R


def sample_signal(R):
    idx = [min(int((j*T_BIT+SAMPLE_TIME)/DT), len(R)-1)
           for j in range(NUM_BITS)]
    return np.array([R[i] for i in idx])



# signal means 
bits = np.random.randint(0, 2, NUM_BITS)

C_clean = build_fick_channel(bits)
R_det = solve_tm_ode(C_clean)
samples = sample_signal(R_det)

mu_1 = np.mean(samples[bits == 1])
mu_0 = np.mean(samples[bits == 0])

print(f"mu1 = {mu_1:.4f}")
print(f"mu0 = {mu_0:.4f}")


# BEr and optimal threshold

def optimal_threshold(mu1, mu0, s1, s0):
    def f(theta):
        return (theta-mu0)/(s0**2) - (mu1-theta)/(s1**2)

    return brentq(f, mu0, mu1)


analytical = []
simulated = []

print("\nN  | Analyt. BER | Sim BER | Error %")


for N in N_RECEPTORS_LIST:

    # Variances
    sigma1 = np.sqrt(mu_1*(1-mu_1)/N)
    sigma0 = np.sqrt(mu_0*(1-mu_0)/N)

    # Optimal threshold (numerical)
    theta = optimal_threshold(mu_1, mu_0, sigma1, sigma0)

    # Analytical BER
    ber_a = 0.5*q_func((mu_1-theta)/sigma1) + \
            0.5*q_func((theta-mu_0)/sigma0)

    analytical.append(ber_a)

    # Monte Carlo
    ber_mc = 0
    for _ in range(NUM_ITER):

        C_noisy = np.random.poisson(C_clean*POISSON_SCALE)/POISSON_SCALE
        R_under = solve_tm_ode(C_noisy)

        R_noisy = np.random.binomial(N, R_under)/N
        s = sample_signal(R_noisy)

        dec = (s > theta).astype(int)
        ber_mc += np.mean(dec != bits)

    ber_mc /= NUM_ITER
    simulated.append(ber_mc)

    err = abs(ber_mc - ber_a) / ber_mc * 100
    print(f"{N:<4}| {ber_a:.5f}    | {ber_mc:.5f} | {err:.2f}")


# plot

plt.figure(figsize=(8,6))

plt.plot(N_RECEPTORS_LIST, analytical, 'k-', label="Semi-Analytical")
plt.plot(N_RECEPTORS_LIST, simulated, 'ro', label="Monte Carlo")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of Receptors (N)")
plt.ylabel("BER")
plt.title("Semi-Analytical BER Validation")

plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.savefig("final_ber_validation.pdf")
plt.show()
