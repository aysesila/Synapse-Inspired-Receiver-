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

T_BIT = 0.2
SCALES = [10, 20, 40, 80, 160, 320]

# fickian channel
def build_fick_channel(bits):
    t_total = len(bits) * T_BIT
    t = np.arange(0, t_total, DT)
    C = np.zeros(len(t))

    # Dynamic pulse duration
    pulse_t = np.arange(0, max(2.0, 10*T_BIT), DT)
    pulse_t = np.maximum(pulse_t, 1e-12)

    pulse = (Q_MOLS / ((4*np.pi*D_COEFF*pulse_t)**1.5)) \
            * np.exp(-(D_DIST**2)/(4*D_COEFF*pulse_t))
    pulse *= 0.005

    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT)
            e = min(s + len(pulse), len(C))
            C[s:e] += pulse[:e-s]

    return C

#receivers

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

            R[i] = np.clip(R[i-1] + dR, 0, 1)
            D[i] = np.clip(D[i-1] + dD, 0, 1)

            s = R[i] + D[i]
            if s > 1:
                R[i] /= s
                D[i] /= s

            F[i] = 1 - R[i] - D[i]

    return R

# samplinfs

def sample(R, bits):
    sample_time = min(0.08, T_BIT - DT)  
    idx = [
        min(int((i*T_BIT + sample_time)/DT), len(R)-1)
        for i in range(len(bits))
    ]
    return np.array([R[i] for i in idx])

# fixedthreshold

def detect(samples, bits):
    tr_s, te_s = samples[:TRAIN_BITS], samples[TRAIN_BITS:]
    tr_b, te_b = bits[:TRAIN_BITS], bits[TRAIN_BITS:]

    mu1 = np.mean(tr_s[tr_b == 1]) if np.any(tr_b == 1) else 0
    mu0 = np.mean(tr_s[tr_b == 0]) if np.any(tr_b == 0) else 0

    th = (mu1 + mu0) / 2
    pred = (te_s > th).astype(int)

    return np.mean(pred != te_b)


def estimate_h(K):
    bits = np.zeros(K+1)
    bits[0] = 1

    C = build_fick_channel(bits)
    R = receiver(C, "standard")
    s = sample(R, bits)

    return s

# adaptive threshold
def adaptive_detection(sig, bits, h, K):
    tr_s, te_s = sig[:TRAIN_BITS], sig[TRAIN_BITS:]
    tr_b, te_b = bits[:TRAIN_BITS], bits[TRAIN_BITS:]

    mu1 = np.mean(tr_s[tr_b == 1]) if np.any(tr_b == 1) else h[0]
    mu0 = np.mean(tr_s[tr_b == 0]) if np.any(tr_b == 0) else 0
    base_th = (mu1 + mu0) / 2.0

    #i maded learning damping here not a fixed value it learns from data
    
    isi_vals = []
    residuals = []

    for i in range(K, TRAIN_BITS):
        past_bits = tr_b[i-K:i][::-1]
        isi_est = np.sum(past_bits * h[1:K+1])
        isi_vals.append(isi_est)
        residuals.append(tr_s[i] - base_th)

    isi_vals = np.array(isi_vals)
    residuals = np.array(residuals)

    if len(isi_vals) > 0:
        alpha = np.dot(isi_vals, residuals) / (np.dot(isi_vals, isi_vals) + 1e-12)
        damping = np.clip(alpha, 0.0, 1.0)
    else:
        damping = 0.5

    # detection
    out = []
    b_hat = np.array(tr_b[-K:])[::-1] if K > 0 else np.zeros(0)

    for val in te_s:
        isi = np.sum(b_hat * h[1:K+1]) * damping
        th = base_th + isi

        d = 1 if val > th else 0
        out.append(d)

        b_hat = np.roll(b_hat, 1)
        b_hat[0] = d

    return np.mean(np.array(out) != te_b)

# simülation part

labels = [
    "Baseline",
    "TM Feedback",
    "Adaptive K=3",
    "Adaptive K=5"
]

results = {l: [] for l in labels}
std_results = {l: [] for l in labels}

print("Estimating CSI...")
h3 = estimate_h(3)
h5 = estimate_h(5)

print("Running Monte Carlo...")

for scale in SCALES:
    acc = {l: [] for l in labels}

    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)

        C = build_fick_channel(bits)
        Cn = np.random.poisson(C * scale) / scale

        R_std = receiver(Cn, "standard")
        R_tm  = receiver(Cn, "feedback")

        sig_std = sample(R_std, bits)
        sig_tm  = sample(R_tm, bits)

        acc["Baseline"].append(detect(sig_std, bits))
        acc["TM Feedback"].append(detect(sig_tm, bits))

        acc["Adaptive K=3"].append(adaptive_detection(sig_std, bits, h3, 3))
        acc["Adaptive K=5"].append(adaptive_detection(sig_std, bits, h5, 5))

    for k in labels:
        results[k].append(np.mean(acc[k]))
        std_results[k].append(np.std(acc[k]))

    print(f"Scale {scale} done")

# plot part

plt.figure(figsize=(10, 6))

for k in labels:
    plt.plot(SCALES,
             np.maximum(results[k], 1e-12),
             marker="o",
             label=k)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Signal Scaling Factor")
plt.ylabel("Bit Error Rate (BER)")
plt.title("Final Comparison: TM vs Adaptive Thresholding")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig("FINAL_READY.pdf")
plt.show()

# table with scale 80

idx = SCALES.index(80)

print("\n=== FINAL TABLE (Scale = 80) ===")
for k in labels:
    print(f"{k:<20}: BER = {results[k][idx]:.5f} ± {std_results[k][idx]:.5f}")
