import numpy as np
import matplotlib.pyplot as plt


dt = 0.001          
T_bit = 1.0          
k_on = 50.0          
k_off = 10.0         
k_rec = 0.2          

def create_rectangular_pulse(t, start_time, duration=0.9):
    
    return np.where((t >= start_time) & (t < start_time + duration), 1.0, 0.0)


# Genereate standart receiver 

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

 # Genereate our TM receiver 
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

#run experiment

def run_rect_experiment(bits, scenario_name, custom_pulse=None):
   
    if custom_pulse:
        total_time = 5.0
        time = np.arange(0, total_time, dt)
        concentration = np.where((time >= custom_pulse[0]) & (time < custom_pulse[1]), 1.0, 0.0)
    else:
        total_time = len(bits) * T_bit
        time = np.arange(0, total_time, dt)
        concentration = np.zeros(len(time))
        for i, bit in enumerate(bits):
            if bit == 1:
                concentration += create_rectangular_pulse(time, i * T_bit + 0.05)
    
    # Add minimal Gaussian noise
    concentration += 0.005 * np.random.normal(size=len(time))
    concentration[concentration < 0] = 0
    
    # Execute Models
    out_std = simulate_standard(concentration)
    out_tm = simulate_tm(concentration)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    def add_grid_and_labels(ax, is_custom):
        if is_custom:
            ax.axvline(x=custom_pulse[0], color='k', linestyle=':', alpha=0.5)
            ax.axvline(x=custom_pulse[1], color='k', linestyle=':', alpha=0.5)
        else:
            for i in range(len(bits) + 1):
                ax.axvline(x=i*T_bit, color='black', linestyle=':', alpha=0.5)
            for i, bit in enumerate(bits):
                ax.text(i*T_bit + 0.5, ax.get_ylim()[1]*0.85, f"Bit {bit}", 
                        ha='center', fontweight='bold', color='darkblue')

    # Plot 1: Input
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time, concentration, 'k', alpha=0.7)
    ax1.fill_between(time, concentration, color='gray', alpha=0.2)
    ax1.set_title(f"1. Rectangular Input: {bits if not custom_pulse else 'Long Pulse'} ({scenario_name})")
    ax1.set_ylabel("Concentration")
    add_grid_and_labels(ax1, custom_pulse is not None)
    
    # Plot 2: Standard
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time, out_std, 'r--', linewidth=2)
    ax2.set_title("2. Standard Receiver (ISI Saturation)")
    ax2.set_ylabel("Active Receptors")
    add_grid_and_labels(ax2, custom_pulse is not None)
    
    # Plot 3: TM
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time, out_tm, 'b-', linewidth=2)
    ax3.set_title(f"3. TM Receiver: Edge Detection (k_rec={k_rec})")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Active Receptors")
    add_grid_and_labels(ax3, custom_pulse is not None)
    
    plt.tight_layout()

# run test scenario

# Scenario 1: Long Rectangular Pulse (from single_pulse.py)
run_rect_experiment(None, "Long Pulse Test", custom_pulse=(0.5, 3.5))

# Scenario 2: Standard Bit Sequences (from high_rate_seq_rect.py)
test_cases = [
    ([1, 0, 1, 0], "Alternating Sequence"),
    ([1, 1, 0, 0], "Consecutive Bits - Depression Effect"),
    ([1, 0, 0, 1], "Recovery Duration Test")
]

for bit_seq, name in test_cases:
    run_rect_experiment(bit_seq, name)

plt.show()
