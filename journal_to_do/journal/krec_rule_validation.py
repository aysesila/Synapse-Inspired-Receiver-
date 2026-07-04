# dosya adı: krec_rule_validation.py
# Bu bölümün (System Model - Recovery-Rate Selection) figürünü üretir.
# Ciktilar: krec_rule_validation.png  ve  krec_rule_validation.pdf

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# ---- simulation parameters (same as the paper) ----
DT = 0.005
T_BIT = 0.2
K_ON = 50.0
K_OFF = 5.0
Q = 50000
SCALE = 80
SAMPLE_TIME = 0.08
NUM_BITS = 900
TRAIN = 120
ITER = 8

COEFF = 0.8   # rule under test: krec = COEFF * (D / d^2)

GRID = np.array([0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.4, 3.0, 3.8, 4.8, 6.0])
CHANNELS = [(3, 30), (3.5, 30), (4, 30), (4.5, 30),
            (4, 40), (4, 60), (4, 80)]


def simulate_static_tm(C, k_rec):
    n = len(C)
    R = np.zeros(n); D = np.zeros(n); F = np.ones(n)
    for i in range(1, n):
        R[i] = min(1.0, max(0.0, R[i-1] + (K_ON*C[i-1]*F[i-1] - K_OFF*R[i-1])*DT))
        D[i] = min(1.0, max(0.0, D[i-1] + (K_OFF*R[i-1] - k_rec*D[i-1])*DT))
        tot = R[i] + D[i]
        if tot > 1.0:
            R[i] /= tot; D[i] /= tot
        F[i] = min(1.0, max(0.0, 1.0 - R[i] - D[i]))
    return R


def build_fick_channel(bits, d, Dc):
    n = int(NUM_BITS * T_BIT / DT)
    C = np.zeros(n)
    pt = np.maximum(np.arange(0, 2.0, DT), 1e-12)
    pulse = (Q / ((4*np.pi*Dc*pt)**1.5)) * np.exp(-(d**2)/(4*Dc*pt))
    pulse *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT)
            e = min(s + len(pulse), n)
            C[s:e] += pulse[:e-s]
    return C


def compute_ber(R, bits):
    idx = [min(int((j*T_BIT + SAMPLE_TIME)/DT), len(R)-1) for j in range(NUM_BITS)]
    s = np.array([R[i] for i in idx])
    mu1 = np.mean(s[:TRAIN][bits[:TRAIN] == 1])
    mu0 = np.mean(s[:TRAIN][bits[:TRAIN] == 0])
    th = (mu1 + mu0) / 2.0
    return np.mean((s[TRAIN:] > th).astype(int) != bits[TRAIN:])


def averaged_ber(d, Dc, k_rec):
    vals = []
    for _ in range(ITER):
        bits = np.random.randint(0, 2, NUM_BITS)
        Cn = np.random.poisson(build_fick_channel(bits, d, Dc) * SCALE) / SCALE
        vals.append(compute_ber(simulate_static_tm(Cn, k_rec), bits))
    return np.mean(vals)


# ---- BER-vs-krec curve for each channel ----
print("Computing BER(krec) curves...\n")
print(f"{'d':>4} {'D':>4} | {'D/d^2':>6} | {'rule k':>7} {'BER@rule':>9} | "
      f"{'best k':>7} {'BER_best':>9} | {'penalty':>7} | {'kopt/x':>6}")
print("-" * 84)

records = []
for (d, Dc) in CHANNELS:
    x = Dc / d**2
    curve = np.array([averaged_ber(d, Dc, k) for k in GRID])
    best_ber = curve.min()
    best_k = GRID[int(np.argmin(curve))]
    k_rule = COEFF * x
    ber_rule = float(np.interp(k_rule, GRID, curve))
    penalty = ber_rule / max(best_ber, 5e-4)
    records.append(dict(d=d, Dc=Dc, x=x, curve=curve,
                        best_ber=best_ber, best_k=best_k,
                        k_rule=k_rule, ber_rule=ber_rule, penalty=penalty))
    print(f"{d:>4} {Dc:>4} | {x:6.2f} | {k_rule:7.2f} {ber_rule:9.4f} | "
          f"{best_k:7.2f} {best_ber:9.4f} | {penalty:6.2f}x | {best_k/x:6.2f}")

BER_FLOOR = 5e-4
good = [r for r in records if BER_FLOOR < r["best_ber"] < 0.15]
ratios = [r["best_k"]/r["x"] for r in good]
pens = [r["penalty"] for r in good]

print("\n" + "=" * 84)
print(f"krec_opt / (D/d^2) ratio (well-resolved): "
      f"min {min(ratios):.2f}, max {max(ratios):.2f}, median {np.median(ratios):.2f}")
print(f"BER penalty of rule (krec={COEFF}*D/d^2): "
      f"median {np.median(pens):.2f}x, max {max(pens):.2f}x")
print("=" * 84)

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(good)))

for r, col in zip(good, colors):
    ax1.plot(GRID, r["curve"], "-", color=col, lw=2.0,
             label=f"d={r['d']}, D={r['Dc']} (D/d²={r['x']:.1f})")
    ax1.plot(r["k_rule"], r["ber_rule"], "o", color=col, ms=10,
             markeredgecolor="k", zorder=5)
ax1.set_yscale("log")
ax1.set_xlabel(r"$k_{rec}$  (1/s)")
ax1.set_ylabel("Average BER")
ax1.set_title(r"BER valley vs $k_{rec}$ (markers = rule $0.8\,D/d^2$)")
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend(fontsize=8, loc="upper right")

for r, col in zip(good, colors):
    ax2.plot(GRID / r["x"], r["curve"] / max(r["best_ber"], BER_FLOOR),
             "-", color=col, lw=2.0, label=f"D/d²={r['x']:.1f}")
ax2.axvline(COEFF, color="green", ls="--", lw=2.0, label=f"rule c={COEFF}")
ax2.set_xlim(0, 3); ax2.set_ylim(0.8, 20); ax2.set_yscale("log")
ax2.set_xlabel(r"$k_{rec} / (D/d^2)$")
ax2.set_ylabel(r"BER / BER$_{\min}$")
ax2.set_title("Collapse test: minima cluster near the rule")
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("krec_rule_validation.png", dpi=200)
plt.savefig("krec_rule_validation.pdf")
plt.show()
