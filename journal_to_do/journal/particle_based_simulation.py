# file: particle_based_simulation.py
# #8 - Channel-model validation: particle-based mean concentration vs ODE Fickian
# Outputs: particle_validation.png , .pdf
# NOTE: particle sim is slow; increase N_PARTICLES / N_RUNS for a smoother curve.

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.titlesize": 15,
    "axes.titleweight": "bold", "lines.linewidth": 2.3, "figure.dpi": 130,
})
np.random.seed(3)

DT = 0.005
D_COEFF = 30.0      # diffusion coefficient (um^2/s)
D_DIST = 4.0        # Tx-Rx distance (um)
R_RX = 2.0          # receiver radius (um)
N_PARTICLES = 8000  # molecules released per run
N_RUNS = 12         # averaged runs (reduces particle noise)
TMAX = 1.5          # observation window (s)

n = int(TMAX / DT)
tgrid = np.arange(n) * DT

# --- ODE Fickian single-pulse response (theory) ---
pt = np.maximum(tgrid, 1e-12)
ode = (1.0 / ((4*np.pi*D_COEFF*pt)**1.5)) * np.exp(-(D_DIST**2)/(4*D_COEFF*pt))

# --- particle-based mean response ---
print("Particle simulation (slow)...")
counts = np.zeros(n)
sigma = np.sqrt(2*D_COEFF*DT)
for r in range(N_RUNS):
    pos = np.zeros((N_PARTICLES, 3)); pos[:, 0] = D_DIST
    for step in range(n):
        pos += np.random.normal(0, sigma, pos.shape)
        counts[step] += np.sum(np.sum(pos**2, axis=1) <= R_RX**2)
    print(f"  run {r+1}/{N_RUNS}")
counts /= N_RUNS

# --- normalize both to peak = 1 (shape comparison) ---
ode_n = ode / np.max(ode)
part_n = counts / max(np.max(counts), 1e-9)

tp_ode = tgrid[np.argmax(ode_n)]
tp_part = tgrid[np.argmax(part_n)]
tp_theory = D_DIST**2 / (6*D_COEFF)
mask = tgrid <= 1.0
nmse = np.mean((ode_n[mask]-part_n[mask])**2) / np.mean(ode_n[mask]**2)

print(f"\nODE peak time      = {tp_ode:.3f} s")
print(f"Particle peak time = {tp_part:.3f} s")
print(f"Theory d^2/6D      = {tp_theory:.3f} s")
print(f"Shape NMSE (t<=1s) = {nmse:.4f}")

# --- plot ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(tgrid, ode_n, "-", color="#c0392b", lw=2.6, label="ODE (Fickian, theory)")
ax.plot(tgrid, part_n, "o", color="#2e86de", ms=4, markevery=6,
        label="Particle-based simulation")
ax.axvline(tp_theory, color="k", ls=":", lw=1.4,
           label=f"$t_{{peak}}=d^2/6D={tp_theory:.3f}$ s")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Normalized concentration at receiver")
ax.set_title(f"Particle-Based Validation of the Fickian Channel Model\n"
             f"(shape NMSE $=$ {nmse:.3f})")
ax.set_xlim(0, TMAX)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("particle_validation.png", dpi=200)
plt.savefig("particle_validation.pdf")
plt.show()
