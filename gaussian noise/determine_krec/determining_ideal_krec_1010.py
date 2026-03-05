import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & SCENARIO
# ==========================================
dt = 0.001
T_bit = 1.0
bits = [1, 0, 1, 0]  # SCENARIO: 1-0-1-0 (Recovery Gap Test)
total_time = len(bits) * T_bit
time = np.arange(0, total_time, dt)

# ==========================================
# 2. INPUT SIGNAL GENERATION
# ==========================================
concentration = np.zeros(len(time))

def create_pulse(t, start_time):
    # Standard diffusion tail (decay=0.2)
    return np.exp(-(t - start_time) / 0.2) * (t > start_time)

for i, bit in enumerate(bits):
    if bit == 1:
        concentration += create_pulse(time, i * T_bit + 0.1)

# Minimal noise
concentration += 0.01 * np.random.normal(size=len(time))
concentration[concentration < 0] = 0

# ==========================================
# 3. TM RECEIVER MODEL (FUNCTION)
# ==========================================
def simulate_tm(C, k_on, k_off, k_rec, total=1.0):
    R_bound = np.zeros(len(C)); R_depressed = np.zeros(len(C)); F = np.zeros(len(C))
    F[0] = total
    
    for i in range(1, len(C)):
        binding = k_on * C[i-1] * F[i-1]
        unbinding = k_off * R_bound[i-1]
        recovery = k_rec * R_depressed[i-1]
        
        d_bound = (binding - unbinding) * dt
        d_depressed = (unbinding - recovery) * dt
        
        R_bound[i] = R_bound[i-1] + d_bound
        R_depressed[i] = R_depressed[i-1] + d_depressed
        F[i] = total - R_bound[i] - R_depressed[i]
    return R_bound

# ==========================================
# 4. EXECUTION (The 3 Candidates)
# ==========================================
k_on = 50.0
k_off = 10.0

# THREE CANDIDATES:
k_slow = 0.2   # Strict Filtering
k_sweet = 0.45 # SWEET SPOT (Balanced)
k_fast = 1.5   # Too Fast

out_slow = simulate_tm(concentration, k_on, k_off, k_slow)
out_sweet = simulate_tm(concentration, k_on, k_off, k_sweet)
out_fast = simulate_tm(concentration, k_on, k_off, k_fast)

# ==========================================
# 5. VISUALIZATION (Comparison)
# ==========================================
plt.figure(figsize=(12, 12))

# Helper to draw bit boundaries
def add_grid(ax):
    for i in range(len(bits) + 1):
        ax.axvline(x=i*T_bit, color='black', linestyle=':', alpha=0.4)
    # Label bits
    for i, bit in enumerate(bits):
        ax.text(i*T_bit + 0.5, ax.get_ylim()[1]*0.9, f"Bit '{bit}'", 
                ha='center', color='darkblue', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# 1. INPUT
ax1 = plt.subplot(4, 1, 1)
ax1.plot(time, concentration, 'k', alpha=0.5)
ax1.fill_between(time, concentration, color='gray', alpha=0.3)
ax1.set_title(f"1. Input Signal: {bits} (Tails invade the '0' slots)", fontsize=12)
ax1.set_ylabel("Conc.")
add_grid(ax1)

# 2. SLOW (0.2)
ax2 = plt.subplot(4, 1, 2)
ax2.plot(time, out_slow, 'b-', linewidth=2)
ax2.set_title(f"2. Strict Filtering (k_rec={k_slow})6 ", fontsize=12)
ax2.set_ylabel("Active Receptors")
add_grid(ax2)

# 3. SWEET SPOT (0.45) <-- LOOK AT THIS
ax3 = plt.subplot(4, 1, 3)
ax3.plot(time, out_sweet, 'g-', linewidth=3)
ax3.set_title(f"3. Sweet Spot (k_rec={k_sweet}) ", fontsize=12, fontweight='bold')
ax3.set_ylabel("Active Receptors")
add_grid(ax3)

# 4. FAST (1.5)
ax4 = plt.subplot(4, 1, 4)
ax4.plot(time, out_fast, 'r-', linewidth=2)
ax4.set_title(f"4. Too Fast (k_rec={k_fast})", fontsize=12)
ax4.set_xlabel("Time (seconds)")
ax4.set_ylabel("Active Receptors")
add_grid(ax4)

plt.tight_layout()
plt.show()
