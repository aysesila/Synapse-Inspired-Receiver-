
# Static TM krec = 0.8*D/d^2 = 1.5 (nominal kanal)


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 15, "axes.titleweight": "bold",
    "lines.linewidth": 2.3, "figure.dpi": 140
})
np.random.seed(42)

DT = 0.005; K_ON = 50.0; K_OFF = 5.0
NUM_BITS = 3000; TRAIN_BITS = 200; NUM_ITERATIONS = 15
Q_MOLS = 50000; D_COEFF = 30.0; D_DIST = 4.0
T_BIT = 0.2; SCALES = [10, 20, 40, 80, 160, 320]
K_REC_STATIC = 0.8 * D_COEFF / (D_DIST ** 2)   # 1.5

def build_fick_channel(bits):
    t = np.arange(0, len(bits)*T_BIT, DT); C = np.zeros(len(t))
    pt = np.maximum(np.arange(0, max(2.0, 10*T_BIT), DT), 1e-12)
    pulse = (Q_MOLS/((4*np.pi*D_COEFF*pt)**1.5))*np.exp(-(D_DIST**2)/(4*D_COEFF*pt)); pulse *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT); e = min(s+len(pulse), len(C)); C[s:e] += pulse[:e-s]
    return C

def receiver(C, mode="standard"):
    R = np.zeros(len(C)); D = np.zeros(len(C)); F = np.ones(len(C))
    for i in range(1, len(C)):
        if mode == "standard":
            R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1); F[i] = 1-R[i]
        else:
            if mode == "static": k = K_REC_STATIC
            elif mode == "feedback": k = 0.2 + 3.5*D[i-1]
            R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1)
            D[i] = np.clip(D[i-1] + (K_OFF*R[i-1]-k*D[i-1])*DT, 0, 1)
            s = R[i]+D[i]
            if s > 1: R[i] /= s; D[i] /= s
            F[i] = 1-R[i]-D[i]
    return R

def sample(R, bits):
    st = min(0.08, T_BIT-DT)
    idx = [min(int((i*T_BIT+st)/DT), len(R)-1) for i in range(len(bits))]
    return np.array([R[i] for i in idx])

def detect(samples, bits):
    ts, es = samples[:TRAIN_BITS], samples[TRAIN_BITS:]; tb, eb = bits[:TRAIN_BITS], bits[TRAIN_BITS:]
    mu1 = np.mean(ts[tb==1]) if np.any(tb==1) else 0; mu0 = np.mean(ts[tb==0]) if np.any(tb==0) else 0
    th = (mu1+mu0)/2
    return np.mean((es>th).astype(int) != eb)

def estimate_h(K):
    bits = np.zeros(K+1); bits[0] = 1
    return sample(receiver(build_fick_channel(bits), "standard"), bits)

def adaptive_detection(sig, bits, h, K):
    ts, es = sig[:TRAIN_BITS], sig[TRAIN_BITS:]; tb, eb = bits[:TRAIN_BITS], bits[TRAIN_BITS:]
    mu1 = np.mean(ts[tb==1]) if np.any(tb==1) else h[0]; mu0 = np.mean(ts[tb==0]) if np.any(tb==0) else 0
    base_th = (mu1+mu0)/2.0
    iv, rs = [], []
    for i in range(K, TRAIN_BITS):
        iv.append(np.sum(tb[i-K:i][::-1]*h[1:K+1])); rs.append(ts[i]-base_th)
    iv = np.array(iv); rs = np.array(rs)
    damping = np.clip(np.dot(iv, rs)/(np.dot(iv, iv)+1e-12), 0.0, 1.0) if len(iv) > 0 else 0.5
    out = []; b_hat = np.array(tb[-K:])[::-1] if K > 0 else np.zeros(0)
    for val in es:
        isi = np.sum(b_hat*h[1:K+1])*damping; th = base_th+isi
        d = 1 if val > th else 0; out.append(d); b_hat = np.roll(b_hat, 1); b_hat[0] = d
    return np.mean(np.array(out) != eb)

labels = ["Baseline (no krec)", f"Static TM (krec=0.8 D/d²={K_REC_STATIC:.1f})",
          "TM Feedback (dynamic krec)", "Adaptive K=3", "Adaptive K=5"]
results = {l: [] for l in labels}
print(f"Rule krec = {K_REC_STATIC:.2f}\nEstimating CSI...")
h3 = estimate_h(3); h5 = estimate_h(5)
print("Running Monte Carlo...")
for scale in SCALES:
    acc = {l: [] for l in labels}
    for _ in range(NUM_ITERATIONS):
        bits = np.random.randint(0, 2, NUM_BITS)
        Cn = np.random.poisson(build_fick_channel(bits)*scale)/scale
        acc[labels[0]].append(detect(sample(receiver(Cn, "standard"), bits), bits))
        acc[labels[1]].append(detect(sample(receiver(Cn, "static"), bits), bits))
        acc[labels[2]].append(detect(sample(receiver(Cn, "feedback"), bits), bits))
        acc[labels[3]].append(adaptive_detection(sample(receiver(Cn, "standard"), bits), bits, h3, 3))
        acc[labels[4]].append(adaptive_detection(sample(receiver(Cn, "standard"), bits), bits, h5, 5))
    for k in labels: results[k].append(np.mean(acc[k]))
    print(f"Scale {scale} done")

plt.figure(figsize=(10, 6))
for k in labels: plt.plot(SCALES, np.maximum(results[k], 1e-12), marker="o", label=k)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Signal Scaling Factor"); plt.ylabel("Bit Error Rate (BER)")
plt.title("TM (with / without krec) vs Adaptive Thresholding")
plt.grid(True, which="both", linestyle="--", alpha=0.4); plt.legend()
plt.tight_layout()
plt.savefig("tm_vs_adaptive_v2.pdf"); plt.savefig("tm_vs_adaptive_v2.png", dpi=200)
plt.show()

idx = SCALES.index(80)
print("\n=== Scale=80 ===")
for k in labels: print(f"{k:<32}: BER={results[k][idx]:.4f}")
