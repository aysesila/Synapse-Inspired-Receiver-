import numpy as np
import matplotlib.pyplot as plt


dt = 0.001
T_bit = 1.0          
k_on = 50.0          # Binding rate
k_off = 10.0         # Unbinding rate
k_rec = 0.45          # Recovery rate

#C=concentration , F= free receptor, R=active receptor, D=Depressed Receptor

#Generate exp. decaying input signal

def create_pulse(t, start_time):
    return np.exp(-(t - start_time) / 0.2) * (t > start_time)

# Generate standart LR receiver
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

# Generate our 3- state TM depleted receiver
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

# Run experiment 
def run_experiment(bits, scenario_name):
    total_time = len(bits) * T_bit
    time = np.arange(0, total_time + 0.5, dt) # Add a little extra time
    concentration = np.zeros(len(time))

    for i, bit in enumerate(bits):
        if bit == 1:
            concentration += create_pulse(time, i * T_bit + 0.1)

    
    concentration += 0.01 * np.random.normal(size=len(time))#Gaussian noise to simulate Brownian motion
    concentration[concentration < 0] = 0

    # Execute Simulations
    out_std = simulate_standard(concentration)
    out_tm = simulate_tm(concentration)

    plt.figure(figsize=(10, 8))
    
    # Subplot 1: Channel Concentration
    plt.subplot(3, 1, 1)
    plt.plot(time, concentration, 'k', alpha=0.6)
    plt.fill_between(time, concentration, color='gray', alpha=0.2)
    plt.title(f"Input Signal: {bits} ({scenario_name})")
    plt.ylabel("Concentration")
    
    # Subplot 2: Standard Receiver 
    plt.subplot(3, 1, 2)
    plt.plot(time, out_std, 'r--', linewidth=2)
    plt.title("Standard Receiver: Fails to clear ISI")
    plt.ylabel("Active Receptors")
    
    # Subplot 3: TM Receiver (Success Case)
    plt.subplot(3, 1, 3)
    plt.plot(time, out_tm, 'b-', linewidth=2)
    plt.title(f"TM Receiver (k_rec={k_rec}): Successfully mitigates ISI")
    plt.ylabel("Active Receptors")
    plt.xlabel("Time (seconds)")
    
    for i in range(len(bits) + 1):
        for plot_idx in [1, 2, 3]:
            plt.subplot(3,1,plot_idx).axvline(x=i*T_bit, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()


# Every test case 
test_cases = [
    ([1], "Single Pulse "),
    ([1, 0], "1-0 sequence"),
    ([1, 1, 0, 0], "Consecutive Bits (Depression Effect)"),
    ([1, 0, 1, 0], "Alternating Bits (High Rate)"),
    ([1, 0, 0, 1], "Recovery Duration Test")
]

for bit_sequence, name in test_cases:
    run_experiment(bit_sequence, name)

plt.show()
