
# Effective throughput vs symbol interval T_bit (3D Fickian)
# throughput = (1 - BER) / T_bit  [bit/s].  Static TM krec = 0.8*D/d^2.
# NUM_ITERATIONS=20, NUM_BITS=2000

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.titlesize": 16,
    "axes.titleweight": "bold", "lines.linewidth": 2.2, "figure.dpi": 130,
})
np.random.seed(42)

DT = 0.005; K_ON = 50.0; K_OFF = 5.0
NUM_BITS = 1500; TRAIN_BITS = 150; NUM_ITERATIONS = 10
SCALE = 80; SAMPLE_FRAC = 0.4     # ornekleme ani = SAMPLE_FRAC * T_bit
Q = 50000; D_COEFF = 30.0; D_DIST = 4.0
KREC_RULE = 0.8 * D_COEFF / (D_DIST ** 2)   # 1.5
T_BITS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]

def build_fick_channel(bits, T_bit):
    n = int(len(bits)*T_bit/DT); C = np.zeros(n)
    pt = np.maximum(np.arange(0, max(2.0, 10*T_bit), DT), 1e-12)
    p = (Q/((4*np.pi*D_COEFF*pt)**1.5))*np.exp(-(D_DIST**2)/(4*D_COEFF*pt)); p *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_bit/DT); e = min(s+len(p), n); C[s:e] += p[:e-s]
    return C

def receiver(C, mode):
    R = np.zeros(len(C)); D = np.zeros(len(C)); F = np.ones(len(C))
    for i in range(1, len(C)):
        if mode == "standard":
            R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1); F[i] = 1-R[i]
        else:
            k = KREC_RULE if mode == "static" else 0.2 + 3.5*D[i-1]
            R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1)
            D[i] = np.clip(D[i-1] + (K_OFF*R[i-1]-k*D[i-1])*DT, 0, 1)
            s = R[i]+D[i]
            if s > 1: R[i] /= s; D[i] /= s
            F[i] = 1-R[i]-D[i]
    return R

def ber(R, bits, T_bit):
    st = min(SAMPLE_FRAC*T_bit, T_bit-DT)
    idx = [min(int((j*T_bit+st)/DT), len(R)-1) for j in range(len(bits))]
    s = np.array([R[i] for i in idx])
    mu1 = np.mean(s[:TRAIN_BITS][bits[:TRAIN_BITS]==1]); mu0 = np.mean(s[:TRAIN_BITS][bits[:TRAIN_BITS]==0])
    th = (mu1+mu0)/2
    return np.mean((s[TRAIN_BITS:]>th).astype(int) != bits[TRAIN_BITS:])

def avg_ber(mode, T_bit):
    vals = []
    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)
        Cn = np.random.poisson(build_fick_channel(bits, T_bit)*SCALE)/SCALE
        vals.append(ber(receiver(Cn, mode), bits, T_bit))
    return np.mean(vals)

modes  = ["standard", "static", "feedback"]
labels = ["Standard 2-State (Baseline)", "Static TM (krec=0.8 D/d²)", "TM Feedback (dynamic)"]
thr = {l: [] for l in labels}
print(f"Rule krec = {KREC_RULE:.2f}")
print("T_bit   | " + " | ".join(f"{l.split('(')[0].strip():<18}" for l in labels))
for T_bit in T_BITS:
    row = []
    for m, lab in zip(modes, labels):
        b = avg_ber(m, T_bit)
        tp = (1.0 - b) / T_bit          # bit/s
        thr[lab].append(tp); row.append(f"BER={b:.3f} tp={tp:5.1f}")
    print(f"{T_bit:.2f}    | " + " | ".join(row))

fig, ax = plt.subplots(figsize=(10, 6.2))
colors = ["#34495e", "#27ae60", "#8e44ad"]
for lab, col in zip(labels, colors):
    ax.plot([1.0/t for t in T_BITS], thr[lab], "o-", color=col, label=lab)
ax.set_xlabel("Symbol Rate $1/T_{bit}$ (symbols/s)")
ax.set_ylabel("Effective Throughput $(1-\\mathrm{BER})/T_{bit}$ (bit/s)")
ax.set_title("Throughput vs Symbol Rate under 3D Fickian Diffusion Channel")
ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=10, loc="best")
plt.tight_layout()
plt.savefig("throughput_analysis.png", dpi=200)
plt.savefig("throughput_analysis.pdf")
plt.show()
