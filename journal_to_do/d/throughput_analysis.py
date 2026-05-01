import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.titleweight": "bold",
    "lines.linewidth": 2.3,
    "figure.dpi": 140
})

np.random.seed(42)

DT = 0.005
K_ON = 50.0
K_OFF = 5.0

NUM_BITS = 3000
TRAIN_BITS = 200
NUM_ITERATIONS = 15

Q_MOLS = 50000
D_COEFF = 30.0
D_DIST = 4.0
POISSON_SCALE = 80.0

T_BIT_LIST = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

#fickian channel 

def build_fick_channel(bits, t_bit):
    t_total = len(bits) * t_bit
    t = np.arange(0, t_total, DT)
    C = np.zeros(len(t))

    pulse_t = np.arange(0, max(2.0, 10*t_bit), DT)
    pulse_t = np.maximum(pulse_t, 1e-12)

    pulse = (Q_MOLS / ((4*np.pi*D_COEFF*pulse_t)**1.5)) \
            * np.exp(-(D_DIST**2)/(4*D_COEFF*pulse_t))
    pulse *= 0.005

    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*t_bit/DT)
            e = min(s+len(pulse), len(C))
            C[s:e] += pulse[:e-s]

    return C

# receivers

def receiver(C, mode="standard"):
    R = np.zeros(len(C))
    D = np.zeros(len(C))
    F = np.ones(len(C))

    for i in range(1, len(C)):
        if mode == "standard":
            dR = (K_ON*C[i-1]*F[i-1] - K_OFF*R[i-1]) * DT
            R[i] = np.clip(R[i-1] + dR, 0, 1)
            F[i] = 1 - R[i]

        else:
            k_rec = 0.2 + 3.5*D[i-1]
            dR = (K_ON*C[i-1]*F[i-1] - K_OFF*R[i-1]) * DT
            dD = (K_OFF*R[i-1] - k_rec*D[i-1]) * DT

            R[i] = np.clip(R[i-1]+dR, 0, 1)
            D[i] = np.clip(D[i-1]+dD, 0, 1)

            s = R[i] + D[i]
            if s > 1:
                R[i] /= s
                D[i] /= s

            F[i] = 1 - R[i] - D[i]

    return R

# sampling

def sample(R, bits, t_bit):
    sample_time = min(0.08, t_bit - DT) 

    idx = [
        min(int((i*t_bit + sample_time)/DT), len(R)-1)
        for i in range(len(bits))
    ]

    return np.array([R[i] for i in idx])

#  detection

def detect(samples, bits):
    tr_s, te_s = samples[:TRAIN_BITS], samples[TRAIN_BITS:]
    tr_b, te_b = bits[:TRAIN_BITS], bits[TRAIN_BITS:]

    mu1 = np.mean(tr_s[tr_b==1]) if np.any(tr_b==1) else 0
    mu0 = np.mean(tr_s[tr_b==0]) if np.any(tr_b==0) else 0

    th = (mu1+mu0)/2
    pred = (te_s>th).astype(int)

    return np.mean(pred!=te_b)

# simulation 
labels = ["Baseline", "TM Feedback"]
ber = {l: [] for l in labels}

for t_bit in T_BIT_LIST:
    acc = {l: 0 for l in labels}

    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)

        C = build_fick_channel(bits, t_bit)
        Cn = np.random.poisson(C*POISSON_SCALE)/POISSON_SCALE

        R1 = receiver(Cn, "standard")
        R2 = receiver(Cn, "feedback")

        s1 = sample(R1, bits, t_bit)
        s2 = sample(R2, bits, t_bit)

        acc["Baseline"] += detect(s1, bits)
        acc["TM Feedback"] += detect(s2, bits)

    for k in labels:
        ber[k].append(acc[k]/NUM_ITERATIONS)
        
    print(f"T_bit = {t_bit:.2f}s tamamlandı.")

# throughput calculation

def bsc_capacity(p):
    if p <= 0.0 or p >= 0.5:
        return 0.0
    return 1.0 + p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)

throughput = {l: [] for l in labels}
for l in labels:
    for i, t_bit in enumerate(T_BIT_LIST):
        tp = bsc_capacity(ber[l][i]) / t_bit
        throughput[l].append(tp)

# plots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#Subplot 1:BER vs T_bit
for k in labels:
    ax1.plot(T_BIT_LIST, np.maximum(ber[k], 1e-12), marker="o", label=k)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Bit Duration, $T_{bit}$ (seconds)")
ax1.set_ylabel("Bit Error Rate (BER)")
ax1.set_title("BER vs. Bit Duration")
ax1.grid(True, which="both", alpha=0.4, linestyle="--")
ax1.legend()

# Subplot 2:Achievable Rate vs T_bit


for k in labels:
    ax2.plot(T_BIT_LIST, throughput[k], marker="s", label=k)

ax2.set_xscale("log")
ax2.set_xlabel("Bit Duration, $T_{bit}$ (seconds)")
ax2.set_ylabel("Achievable Rate (bps)")
ax2.set_title("Achievable Rate vs. Bit Duration")
ax2.grid(True, which="both", alpha=0.4, linestyle="--")
ax2.legend()

plt.tight_layout()
plt.savefig("Throughput_Analysis_BSC.pdf")
plt.show()

# Console output

for l in labels:
    peak_idx = np.argmax(throughput[l])
    print(f"{l}: Max Rate = {throughput[l][peak_idx]:.2f} bps (at T_bit = {T_BIT_LIST[peak_idx]}s)")
