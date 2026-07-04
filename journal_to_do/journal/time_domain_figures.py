

# 9a: receptor response to [1,1,0,0] for different krec values (determining_krec)
# 9b: [1,0,1,0] sequence, standard vs TM receiver (ISI suppression)
# Outputs: determining_krec.png/.pdf , high_rate_1010.png/.pdf

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 13, "axes.titlesize": 14,
    "axes.titleweight": "bold", "lines.linewidth": 2.0, "figure.dpi": 130,
})
np.random.seed(0)

DT = 0.005; T_BIT = 0.2; K_ON = 50.0; K_OFF = 5.0
Q = 50000; D_COEFF = 30.0; D_DIST = 4.0; SCALE = 80
KREC_RULE = 0.8 * D_COEFF / (D_DIST ** 2)   # 1.5

def fick_channel(bits):
    n = int(len(bits)*T_BIT/DT); C = np.zeros(n)
    pt = np.maximum(np.arange(0, 2.0, DT), 1e-12)
    p = (Q/((4*np.pi*D_COEFF*pt)**1.5))*np.exp(-(D_DIST**2)/(4*D_COEFF*pt)); p *= 0.005
    for i, b in enumerate(bits):
        if b == 1:
            s = int(i*T_BIT/DT); e = min(s+len(p), n); C[s:e] += p[:e-s]
    return C, np.arange(n)*DT

def tm_receiver(C, k_rec):
    R = np.zeros(len(C)); D = np.zeros(len(C)); F = np.ones(len(C))
    for i in range(1, len(C)):
        R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1)
        D[i] = np.clip(D[i-1] + (K_OFF*R[i-1]-k_rec*D[i-1])*DT, 0, 1)
        s = R[i]+D[i]
        if s > 1: R[i] /= s; D[i] /= s
        F[i] = 1-R[i]-D[i]
    return R

def std_receiver(C):
    R = np.zeros(len(C)); F = np.ones(len(C))
    for i in range(1, len(C)):
        R[i] = np.clip(R[i-1] + (K_ON*C[i-1]*F[i-1]-K_OFF*R[i-1])*DT, 0, 1); F[i] = 1-R[i]
    return R

# 9a: determining_krec ([1,1,0,0]) 
bits_a = [1, 1, 0, 0]
C_a, t_a = fick_channel(bits_a)
Cn_a = np.random.poisson(C_a*SCALE)/SCALE
krec_cases = [(0.2, "Too slow ($k_{rec}=0.2$)", "tab:blue"),
              (KREC_RULE, f"Rule ($k_{{rec}}=0.8D/d^2={KREC_RULE:.1f}$)", "tab:green"),
              (5.0, "Too fast ($k_{rec}=5.0$)", "tab:red")]

fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
axes[0].fill_between(t_a, Cn_a, color="0.7", alpha=0.6)
axes[0].set_ylabel("Conc."); axes[0].set_title("Input: [1, 1, 0, 0]")
for ax, (kr, lab, col) in zip(axes[1:], krec_cases):
    R = tm_receiver(Cn_a, kr)
    ax.plot(t_a, R, color=col, lw=2.3)
    ax.set_ylabel("Bound $R_b$"); ax.set_title(lab)
    ax.set_ylim(0, max(0.7, R.max()*1.1))
for i in range(len(bits_a)+1):
    for ax in axes: ax.axvline(i*T_BIT, color="k", ls=":", lw=0.7, alpha=0.5)
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("determining_krec.png", dpi=200); plt.savefig("determining_krec.pdf")

#  high-rate 1010 
bits_b = [1, 0, 1, 0]
C_b, t_b = fick_channel(bits_b)
Cn_b = np.random.poisson(C_b*SCALE)/SCALE
R_std = std_receiver(Cn_b)
R_tm  = tm_receiver(Cn_b, KREC_RULE)

fig2, axes2 = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
axes2[0].fill_between(t_b, Cn_b, color="0.7", alpha=0.6)
axes2[0].set_ylabel("Conc."); axes2[0].set_title("Input: [1, 0, 1, 0] (tails invade the '0' slots)")
axes2[1].plot(t_b, R_std, "r--", lw=2.3)
axes2[1].set_ylabel("Bound $R_b$"); axes2[1].set_title("Standard receiver: fails to return to zero (ISI)")
axes2[2].plot(t_b, R_tm, "b-", lw=2.3)
axes2[2].set_ylabel("Bound $R_b$"); axes2[2].set_title(f"TM-depleted receiver: clean 1-0-1-0 ($k_{{rec}}={KREC_RULE:.1f}$)")
for i in range(len(bits_b)+1):
    for ax in axes2: ax.axvline(i*T_BIT, color="k", ls=":", lw=0.7, alpha=0.5)
axes2[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("high_rate_1010.png", dpi=200); plt.savefig("high_rate_1010.pdf")
plt.show()
print(f"Rule krec = {KREC_RULE:.2f}")
print("Generated: determining_krec.pdf , high_rate_1010.pdf")
