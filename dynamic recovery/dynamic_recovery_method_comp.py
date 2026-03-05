import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
time = np.arange(0, 4.0, dt)
bits = [1, 1, 0, 0]
k_on, k_off = 50.0, 10.0

# Clean signal generation
clean_c = np.zeros(len(time))
for i, b in enumerate(bits):
    if b == 1: clean_c[int((i+0.1)/dt):int((i+0.3)/dt)] = 1.0

# Apply Poisson Noise
noisy_c = np.random.poisson(clean_c * 500) / 500

def sim_dynamic_1(C, k_base=0.2, alpha=1.5):
    R, D, F = np.zeros(len(C)), np.zeros(len(C)), np.zeros(len(C))
    F[0] = 1.0
    for i in range(1, len(C)):
        k_dyn = k_base + alpha * R[i-1]
        R[i] = R[i-1] + (k_on*C[i-1]*F[i-1] - k_off*R[i-1])*dt
        D[i] = D[i-1] + (k_off*R[i-1] - k_dyn*D[i-1])*dt
        F[i] = max(0, 1.0 - R[i] - D[i])
    return R

def sim_dynamic_2(C):
    R, D, F = np.zeros(len(C)), np.zeros(len(C)), np.zeros(len(C))
    F[0] = 1.0
    for i in range(1, len(C)):
        k_dyn = R[i-1] / (D[i-1] + 0.1) # Ratiometric Approach
        R[i] = R[i-1] + (k_on*C[i-1]*F[i-1] - k_off*R[i-1])*dt
        D[i] = D[i-1] + (k_off*R[i-1] - k_dyn*D[i-1])*dt
        F[i] = max(0, 1.0 - R[i] - D[i])
    return R

out1 = sim_dynamic_1(noisy_c)
out2 = sim_dynamic_2(noisy_c)

plt.figure(figsize=(10, 6))
plt.plot(time, out1, 'g-', label='Approach 1: Linear ($k_{base} + alpha \cdot R$)')
plt.plot(time, out2, 'r--', label='Approach 2: State-Dependent ($R/D$)')
plt.title("Comparison of Proposed Dynamic k_rec Models")
plt.ylabel("Active Receptors")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
