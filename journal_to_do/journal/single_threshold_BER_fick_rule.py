# dosya adi: single_threshold_BER_fick_rule.py
# #3 - Single-threshold BER vs signal scaling (3D Fickian), krec = 0.8*D/d^2
# Ciktilar: single_threshold_BER_fick_rule.png , .pdf
# Paper icin final: NUM_ITERATIONS=20, NUM_BITS=2000

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.titlesize": 16,
    "axes.titleweight": "bold", "lines.linewidth": 2.2, "figure.dpi": 130,
})
np.random.seed(42)

DT = 0.005; T_BIT = 0.2; K_ON = 50.0; K_OFF = 5.0
NUM_BITS = 1500; TRAIN_BITS = 200; NUM_ITERATIONS = 12
SCALES = [10, 20, 40, 80, 160, 320]; SAMPLE_TIME = 0.08
Q = 50000; D_COEFF = 30.0; D_DIST = 4.0
KREC_RULE = 0.8 * D_COEFF / (D_DIST ** 2)   # 1.5
KREC_OLD = 0.45

def simulate_receiver(C, mode, krec_val=0.45):
    n = len(C); R = np.zeros(n); D = np.zeros(n); F = np.ones(n)
    if mode == "slope":
        kern = np.ones(5)/5.0; Cs = np.convolve(C, kern, mode="same")
        dC = np.zeros(n); dC[1:] = (Cs[1:]-Cs[:-1])/DT
    for i in range(1, n):
        if mode == "standard":
            R[i] = min(1.0, max(0.0, R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT))
            D[i] = 0.0; F[i] = min(1.0, max(0.0, 1.0-R[i])); continue
        if mode in ("static_rule", "static_old"): k = krec_val
        elif mode == "occupancy": k = 0.2 + 5.0*R[i-1]
        elif mode == "feedback":  k = 0.2 + 3.5*D[i-1]
        elif mode == "slope":     k = 0.2 + 0.8*max(0.0, -dC[i-1])
        R[i] = min(1.0, max(0.0, R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT))
        D[i] = min(1.0, max(0.0, D[i-1] + (K_OFF*R[i-1]-k*D[i-1])*DT))
        tot = R[i]+D[i]
        if tot > 1.0: R[i] /= tot; D[i] /= tot
        F[i] = min(1.0, max(0.0, 1.0-R[i]-D[i]))
    return R

def build_fick_channel(bits):
    n = int(NUM_BITS*T_BIT/DT); C = np.zeros(n)
    pt = np.maximum(np.arange(0, 2.0, DT), 1e-12)
    p = (Q/((4*np.pi*D_COEFF*pt)**1.5))*np.exp(-(D_DIST**2)/(4*D_COEFF*pt)); p *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT); e = min(s+len(p), n); C[s:e] += p[:e-s]
    return C

def ber(R, bits):
    idx = [min(int((j*T_BIT+SAMPLE_TIME)/DT), len(R)-1) for j in range(NUM_BITS)]
    s = np.array([R[i] for i in idx])
    mu1 = np.mean(s[:TRAIN_BITS][bits[:TRAIN_BITS]==1]); mu0 = np.mean(s[:TRAIN_BITS][bits[:TRAIN_BITS]==0])
    th = (mu1+mu0)/2.0
    return np.mean((s[TRAIN_BITS:]>th).astype(int) != bits[TRAIN_BITS:])

modes  = ["standard","static_rule","occupancy","feedback","slope","static_old"]
labels = ["Standard 2-State (Baseline)",
          f"Static TM, rule (krec=0.8 D/d²={KREC_RULE:.2f})",
          "Occupancy-Driven (R)","Internal Feedback (D)",
          "Slope-Aware Differential","Static TM (krec=0.45, previous)"]
res = {l: [] for l in labels}
print(f"Rule krec = {KREC_RULE:.2f}\n")
for scale in SCALES:
    acc = {l: 0.0 for l in labels}
    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)
        Cn = np.random.poisson(build_fick_channel(bits)*scale)/scale
        for m, lab in zip(modes, labels):
            kv = KREC_RULE if m=="static_rule" else KREC_OLD
            acc[lab] += ber(simulate_receiver(Cn, m, kv), bits)
    for lab in labels: res[lab].append(acc[lab]/NUM_ITERATIONS)
    print(f"scale {scale} done")

i80 = SCALES.index(80)
print("\n--- BER at scale=80 ---")
for lab in labels: print(f"{lab:<42}: {res[lab][i80]:.4f}")

fig, ax = plt.subplots(figsize=(10.5, 6.3))
colors = ["#34495e","#27ae60","#f39c12","#8e44ad","#c0392b"]
for lab, col in zip(labels[:5], colors): ax.plot(SCALES, res[lab], "o-", color=col, label=lab)
ax.plot(SCALES, res[labels[5]], "s--", color="0.6", lw=1.8, label=labels[5])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Signal Scaling Factor (higher = stronger signal)")
ax.set_ylabel("Average Bit Error Rate (BER)")
ax.set_title("Single-Threshold BER under 3D Fickian Diffusion Channel")
ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9, loc="best")
plt.tight_layout()
plt.savefig("single_threshold_BER_fick_rule.png", dpi=200)
plt.savefig("single_threshold_BER_fick_rule.pdf")
plt.show()
