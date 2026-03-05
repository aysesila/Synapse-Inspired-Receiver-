import numpy as np
import matplotlib.pyplot as plt

dt = 0.001           
T_bit = 1.0          
k_on = 50.0          
k_off = 10.0         
k_rec = 0.45          

#Generate gaussian pulse

def create_gaussian_pulse(t, center_time, sigma=0.12):
       return np.exp(-((t - center_time)**2) / (2 * sigma**2))

#Generate LR Receiver model

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

#Generate our TM receiver

def simulate_tm(C, total=1.0):
    
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


#Run experiment

def run_gaussian_experiment(bits, scenario_name):
    total_time = len(bits) * T_bit
    time = np.arange(0, total_time, dt)
    concentration = np.zeros(len(time))
    
    
    for i, bit in enumerate(bits):
        if bit == 1:
            center = i * T_bit + 0.5 # 0.5s offset
            concentration += create_gaussian_pulse(time, center)
    
    # Add minimal Gaussian noise
    concentration += 0.005 * np.random.normal(size=len(time))
    concentration[concentration < 0] = 0
    
    # Execute Models
    out_std = simulate_standard(concentration)
    out_tm = simulate_tm(concentration)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    def add_grid_and_labels(ax):
        for i in range(len(bits) + 1):
            ax.axvline(x=i*T_bit, color='black', linestyle=':', alpha=0.5)
        for i, bit in enumerate(bits):
            ax.text(i*T_bit + 0.5, ax.get_ylim()[1]*0.85, f"Bit {bit}", 
                    ha='center', fontweight='bold', color='darkblue')

    # Plot 1: Input
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, concentration, 'k', alpha=0.7)
    ax1.set_title(f"1. Gaussian Input: {bits} ({scenario_name})")
    ax1.set_ylabel("Concentration")
    add_grid_and_labels(ax1)
    
    # Plot 2: Standard
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time, out_std, 'r--', linewidth=2)
    ax2.set_title("2. Standard Receiver (ISI Invasion)")
    ax2.set_ylabel("Active Receptors")
    add_grid_and_labels(ax2)
    
    # Plot 3: TM
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time, out_tm, 'b-', linewidth=2)
    ax3.set_title(f"3. TM Receiver (ISI Mitigated, k_rec={k_rec})")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Active Receptors")
    add_grid_and_labels(ax3)
    
    plt.tight_layout()


# test cases to simulate 
test_cases = [
    ([1], "Single Pulse Baseline"),
    ([1, 0, 1, 0], "Alternating Sequence"),
    ([1, 1, 0, 0], "Consecutive Bits (Depression Effect)"),
    ([1, 0, 0, 1], "Recovery Duration Test")
]

for bit_seq, name in test_cases:
    run_gaussian_experiment(bit_seq, name)

plt.show()
