import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & SCENARIO
# ==========================================
dt = 0.001
T_bit = 1.0
bits = [1, 1, 0, 0]  # The best pattern to see both "damping" and "ISI"
total_time = len(bits) * T_bit
time = np.arange(0, total_time, dt)

# ==========================================
# 2. INPUT SIGNAL GENERATION
# ==========================================
concentration = np.zeros(len(time))

def create_pulse(t, start_time):
    # Standard diffusion tail
    return np.exp(-(t - start_time) / 0.2) * (t > start_time)

for i, bit in enumerate(bits):
    if bit == 1:
        concentration += create_pulse(time, i * T_bit + 0.1)

# Minimal noise
concentration += 0.01 * np.random.normal(size=len(time))
concentration[concentration < 0] = 0

# ==========================================
# 3. TM RECEIVER MODEL
# ==========================================
def simulate_tm(C, k_on, k_off, k_rec, total=1.0):
    R = np.zeros(len(C)); D = np.zeros(len(C)); F = np.zeros(len(C)); F[0] = total
    for i in range(1, len(C)):
        binding = k_on * C[i-1] * F[i-1]
        unbinding = k_off * R[i-1]
        recovery = k_rec * D[i-1]
        
        R[i] = R[i-1] + (binding - unbinding) * dt
        D[i] = D[i-1] + (unbinding - recovery) * dt
        F[i] = total - R[i] - D[i]
    return R

# ==========================================
# 4. EXECUTION (THE COMPARISON)
# ==========================================
k_on = 50.0
k_off = 10.0

# THREE CANDIDATES:
k_slow = 0.2   # Too strict (High damping)
k_sweet = 0.45 # THE SWEET SPOT (Balanced)
k_fast = 1.5   # Too fast (ISI returns)

out_slow = simulate_tm(concentration, k_on, k_off, k_slow)
out_sweet = simulate_tm(concentration, k_on, k_off, k_sweet)
out_fast = simulate_tm(concentration, k_on, k_off, k_fast)

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 12))

# Helper to draw bit boundaries
def add_grid(ax):
    for i in range(len(bits) + 1):
        ax.axvline(x=i*T_bit, color='black', linestyle=':', alpha=0.4)

# 1. INPUT
ax1 = plt.subplot(4, 1, 1)
ax1.plot(time, concentration, 'k', alpha=0.5)
ax1.fill_between(time, concentration, color='gray', alpha=0.3)
ax1.set_title(f"1. Input Signal: {bits} (Reference)", fontsize=12)
ax1.set_ylabel("Conc.")
add_grid(ax1)

# 2. SLOW RECOVERY (0.2)
ax2 = plt.subplot(4, 1, 2)
ax2.plot(time, out_slow, 'b-', linewidth=2)
ax2.set_title(f"2. Strict Filtering (k_rec={k_slow}): ", fontsize=12)
ax2.set_ylabel("Output")
add_grid(ax2)

# 3. SWEET SPOT (0.45) <-- BUNA ODAKLAN
ax3 = plt.subplot(4, 1, 3)
ax3.plot(time, out_sweet, 'g-', linewidth=3) # GREEN IS GOOD
ax3.set_title(f"3. Balanced / Sweet Spot (k_rec={k_sweet}):", fontsize=12, fontweight='bold')
ax3.set_ylabel("Output")
add_grid(ax3)

# 4. FAST RECOVERY (1.5)
ax4 = plt.subplot(4, 1, 4)
ax4.plot(time, out_fast, 'r-', linewidth=2)
ax4.set_title(f"4. Too Fast (k_rec={k_fast}): ", fontsize=12)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Output")
add_grid(ax4)

plt.tight_layout()
plt.show()
