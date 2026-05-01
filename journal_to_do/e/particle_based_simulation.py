import numpy as np
import matplotlib.pyplot as plt
import time


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titleweight": "bold",
    "lines.linewidth": 2.3,
    "figure.dpi": 140
})

np.random.seed(42)

DT = 0.005
T_BIT = 0.2

K_ON = 50.0
K_OFF = 5.0
K_REC = 0.45

D_COEFF = 30.0
D_DIST = 4.0
R_RX = 1.0
V_RX = (4.0/3.0) * np.pi * (R_RX**3)

NUM_BITS = 200
TRAIN_BITS = 50
NUM_ITERATIONS = 25

Q_LIST = [10000, 20000, 30000, 40000, 50000]

CULL_RADIUS = 25.0


# receivers

def tm_receiver(C):
    R = np.zeros(len(C))
    D = np.zeros(len(C))
    F = np.ones(len(C))

    for i in range(1, len(C)):
        dR = (K_ON * C[i-1] * F[i-1] - K_OFF * R[i-1]) * DT
        dD = (K_OFF * R[i-1] - K_REC * D[i-1]) * DT

        R[i] = np.clip(R[i-1] + dR, 0, 1)
        D[i] = np.clip(D[i-1] + dD, 0, 1)

        s = R[i] + D[i]
        if s > 1:
            R[i] /= s
            D[i] /= s

        F[i] = 1 - R[i] - D[i]

    return R

# derector

def detect(R, bits):
    idx = [min(int((i*T_BIT + 0.08)/DT), len(R)-1) for i in range(len(bits))]
    samples = np.array([R[i] for i in idx])

    tr_s, te_s = samples[:TRAIN_BITS], samples[TRAIN_BITS:]
    tr_b, te_b = bits[:TRAIN_BITS], bits[TRAIN_BITS:]

    mu1 = np.mean(tr_s[tr_b==1]) if np.any(tr_b==1) else 0
    mu0 = np.mean(tr_s[tr_b==0]) if np.any(tr_b==0) else 0
    th = (mu1 + mu0) / 2

    return np.mean((te_s > th).astype(int) != te_b)

# ode model

def generate_ode_signal(bits, Q):
    t_total = len(bits) * T_BIT
    t = np.arange(0, t_total, DT)
    C = np.zeros(len(t))

    pulse_t = np.arange(0, 2.0, DT)
    pulse_t = np.maximum(pulse_t, 1e-12)

    pulse = (Q / ((4*np.pi*D_COEFF*pulse_t)**1.5)) \
            * np.exp(-(D_DIST**2)/(4*D_COEFF*pulse_t))

    pulse *= 0.005 

    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT)
            e = min(s+len(pulse), len(C))
            C[s:e] += pulse[:e-s]

    molecules = np.random.poisson(C * V_RX)
    return molecules / V_RX

# particle simulation

def generate_particle_signal(bits, Q):
    t_total = len(bits) * T_BIT
    total_steps = int(t_total / DT)
    counts = np.zeros(total_steps)

    active_particles = []
    std_dev = np.sqrt(2 * D_COEFF * DT)
    steps_per_bit = int(T_BIT / DT)

    step_idx = 0
    for b in bits:
        if b == 1:
            p = np.zeros((Q, 3))
            p[:, 0] = D_DIST
            active_particles.append(p)

        for _ in range(steps_per_bit):
            total_in_rx = 0

            for i in range(len(active_particles)):
                part = active_particles[i]
                if len(part) == 0:
                    continue

                part += np.random.normal(0, std_dev, part.shape)

                dist_sq = np.sum(part**2, axis=1)
                total_in_rx += np.sum(dist_sq <= R_RX**2)

                if step_idx % 20 == 0:
                    active_particles[i] = part[dist_sq < CULL_RADIUS**2]

            counts[step_idx] = total_in_rx
            step_idx += 1

    return (counts / V_RX) * 0.005  

# main part

ber_ode = []
ber_particle = []


for Q in Q_LIST:
    acc_ode = []
    acc_part = []

    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)

        R1 = tm_receiver(generate_ode_signal(bits, Q))
        R2 = tm_receiver(generate_particle_signal(bits, Q))

        acc_ode.append(detect(R1, bits))
        acc_part.append(detect(R2, bits))

    m1 = np.mean(acc_ode)
    m2 = np.mean(acc_part)

    ber_ode.append(m1)
    ber_particle.append(m2)

    err = abs(m1 - m2) / max(m1, 1e-12) * 100

    print(f"Q={Q} | ODE={m1:.4f} | Particle={m2:.4f} | Diff={err:.2f}%")

    if err > 15:
        print("there is a problem warning: >15% mismatch")

# Plots

plt.figure(figsize=(9,6))
plt.plot(Q_LIST, ber_ode, 'o-', label="ODE")
plt.plot(Q_LIST, ber_particle, 's--', label="Particle")

plt.yscale("log")
plt.xlabel("Q")
plt.ylabel("BER")
plt.title("ODE vs Particle Validation")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()
