import numpy as np
import matplotlib.pyplot as plt


dt = 0.001
T_bit = 1.0
k_on = 50.0
k_off = 10.0
k_rec = 0.45          
MOLECULE_SCALE = 1000


#Functions 

def apply_poisson_noise(clean_signal):
    scaled = clean_signal * MOLECULE_SCALE
    noisy = np.random.poisson(scaled)
    return noisy / MOLECULE_SCALE

#generate LR receiver

def simulate_standard(C, total=1.0):
    R = np.zeros(len(C));
    F = np.zeros(len(C));
    F[0] = total
    for i in range(1, len(C)):
        binding = k_on * C[i-1] * F[i-1]
        unbinding = k_off * R[i-1]
        R[i] = R[i-1] + (binding - unbinding) * dt
        F[i] = total - R[i]
    return R

#generate our TM  receiver

def simulate_tm(C, k_rec, total=1.0):
    R = np.zeros(len(C));
    D = np.zeros(len(C));
    F = np.zeros(len(C));
    F[0] = total
    for i in range(1, len(C)):
        binding = k_on * C[i-1] * F[i-1]
        unbinding = k_off * R[i-1]
        recovery = k_rec * D[i-1]
        R[i] = R[i-1] + (binding - unbinding) * dt
        D[i] = D[i-1] + (unbinding - recovery) * dt
        F[i] = total - R[i] - D[i]
    return R

def run_gaussian_test(bits, title, is_single=False):
    
    duration = 3.0 if is_single else len(bits) * T_bit
    time = np.arange(0, duration, dt)
    clean_concentration = np.zeros(len(time))
    
    # Generate Gaussian Pulses
    
    sigma = 0.12 # Pulse width
    for i, bit in enumerate(bits):
        if bit == 1:
            center = i * T_bit + 0.5 
            clean_concentration += np.exp(-((time - center)**2) / (2 * sigma**2))
    
    # Apply Poisson Noise
    noisy_concentration = apply_poisson_noise(clean_concentration)
    
    # Run Both Models
    out_std = simulate_standard(noisy_concentration)
    out_tm = simulate_tm(noisy_concentration, k_rec)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Input with Poisson Noise
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, noisy_concentration, 'k', alpha=0.6, linewidth=1)
    ax1.set_title(f"Gaussian Pulse Scenario: {title} | Input with Poisson Noise", fontweight='bold')
    ax1.set_ylabel("Concentration")
    
    # Subplot 2: Standard Receiver
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time, out_std, 'r--', linewidth=2)
    ax2.set_title("Standard Receiver Output")
    ax2.set_ylabel("Active Receptors")
    
    # Subplot 3: TM Receiver (Optimized)
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time, out_tm, 'g-', linewidth=2.5) # Green for Gaussian tests
    ax3.set_title(f"TM Receiver Output (k_rec={k_rec})")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Active Receptors")
    
    for ax in [ax1, ax2, ax3]:
        for t in range(int(duration) + 1):
            ax.axvline(x=t, color='gray', linestyle=':', alpha=0.4)
            
    plt.tight_layout()
    plt.show()

# # execute test scenarios

# 1. Single Gaussian Pulse
run_gaussian_test([1], "Single Pulse", is_single=True)

# 2. Sequence [1, 0, 1, 0]
run_gaussian_test([1, 0, 1, 0], "Sequence [1, 0, 1, 0]")

# 3. Sequence [1, 1, 0, 0]
run_gaussian_test([1, 1, 0, 0], "Sequence [1, 1, 0, 0]")

# 4. Sequence [1, 0, 0, 1]
run_gaussian_test([1, 0, 0, 1], "Sequence [1, 0, 0, 1]")
