import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETTINGS (Fiziksel Kanal Parametreleri) ---
dt, T_bit = 0.005, 0.1
k_on, k_off = 50.0, 5.0
NUM_BITS = 2000
TRAIN_BITS = 200
SCALES = [5, 10, 20, 40, 80, 160]

# --- 2. MODELS ---
def simulate_standard(C):
    R, F = np.zeros(len(C)), np.ones(len(C))
    for i in range(1, len(C)):
        dR = (k_on * C[i-1] * F[i-1] - k_off * R[i-1]) * dt
        R[i] = R[i-1] + dR
        F[i] = max(0, 1.0 - R[i])
    return R

def simulate_tm(C, k_rec_val, is_dynamic=False, alpha=5.0):
    R, D, F = np.zeros(len(C)), np.zeros(len(C)), np.ones(len(C))
    for i in range(1, len(C)):
        k_val = (0.2 + alpha * R[i-1]) if is_dynamic else k_rec_val
        dR = (k_on * C[i-1] * F[i-1] - k_off * R[i-1]) * dt
        dD = (k_off * R[i-1] - k_val * D[i-1]) * dt
        R[i], D[i] = R[i-1] + dR, D[i-1] + dD
        F[i] = max(0, 1.0 - R[i] - D[i])
    return R

# --- 3. SIGNAL (Difüzyon Kanalı) ---
np.random.seed(42)
bits = np.random.randint(0, 2, NUM_BITS)
time = np.arange(0, NUM_BITS * T_bit, dt)
clean_c = np.zeros(len(time))

for i, b in enumerate(bits):
    if b == 1:
        start_idx = int((i * T_bit) / dt)
        t_array = time[start_idx:] - time[start_idx]
        clean_c[start_idx:] += 1.0 * np.exp(-t_array / 0.15) 

def get_responses(R_sig):
    res = []
    for i in range(NUM_BITS):
        idx = int((i * T_bit + 0.08) / dt) 
        res.append(R_sig[idx])
    return np.array(res)

# --- 4. EXECUTION (Single vs Dual Threshold) ---
res_single = { "Standard": [], "k_rec=0.2": [], "k_rec=0.45": [], "Dynamic": [] }
res_dual   = { "Standard": [], "k_rec=0.2": [], "k_rec=0.45": [], "Dynamic": [] }

for s in SCALES:
    noisy = np.random.poisson(clean_c * s) / s
    outs = { "Standard": simulate_standard(noisy), 
             "k_rec=0.2": simulate_tm(noisy, 0.2),
             "k_rec=0.45": simulate_tm(noisy, 0.45), 
             "Dynamic": simulate_tm(noisy, 0, is_dynamic=True) }
    
    for key, R_sig in outs.items():
        resp = get_responses(R_sig)
        
        # Preamble (Eğitim) Verisi ile Eşikleri Belirleme
        train_resp = resp[:TRAIN_BITS]
        train_bits = bits[:TRAIN_BITS]
        
        mean_1 = np.mean(train_resp[train_bits==1])
        mean_0 = np.mean(train_resp[train_bits==0])
        
        # 1. Single Threshold (Orta Nokta)
        th_single = (mean_1 + mean_0) / 2
        
        # 2. Dual Threshold (Histerezis Bandı - %20 Marj)
        margin = (mean_1 - mean_0) * 0.20
        th_high = th_single + margin
        th_low  = th_single - margin
        
        # Test Verisi
        test_resp = resp[TRAIN_BITS:]
        test_bits = bits[TRAIN_BITS:]
        
        # Single Threshold Kararı
        ber_s = np.mean((test_resp > th_single).astype(int) != test_bits)
        res_single[key].append(float(ber_s))
        
        # Dual Threshold Kararı (Durum Makinesi / Histerezis)
        decisions_dual = []
        current_state = 0 # Başlangıç kararı 0
        for val in test_resp:
            if val > th_high:
                current_state = 1
            elif val < th_low:
                current_state = 0
            # Arada kalırsa current_state değişmez!
            decisions_dual.append(current_state)
            
        ber_d = np.mean(np.array(decisions_dual) != test_bits)
        res_dual[key].append(float(ber_d))

# --- 5. DATA OUTPUT ---
print("\n" + "="*50)
print("SIMULATION RESULTS: SINGLE vs DUAL THRESHOLD")
print("="*50)
for model in ["Standard", "k_rec=0.2", "k_rec=0.45", "Dynamic"]:
    print(f"{model} Single BER = {res_single[model]}")
    print(f"{model} Dual   BER = {res_dual[model]}")
    print("-" * 50)

# --- 6. PLOT ---
plt.figure(figsize=(10, 7))

# Sadece Standard ve Dynamic modellerini kıyaslayalım (Grafik çok karışmasın)
plt.plot(SCALES, [max(v, 1e-4) for v in res_single["Dynamic"]], 'rs-', label="Dynamic (Single Th.)", linewidth=2)
plt.plot(SCALES, [max(v, 1e-4) for v in res_dual["Dynamic"]], 'ro--', label="Dynamic (Dual Th. - Proposed)", linewidth=3)

plt.plot(SCALES, [max(v, 1e-4) for v in res_single["Standard"]], 'ks-', label="Standard (Single Th.)", alpha=0.6)
plt.plot(SCALES, [max(v, 1e-4) for v in res_dual["Standard"]], 'ko--', label="Standard (Dual Th.)", alpha=0.6)

plt.yscale('log'); plt.xscale('log'); plt.grid(True, which="both", alpha=0.3)
plt.title("BER Performance Boost: Single vs Dual Thresholding", fontweight="bold")
plt.xlabel("Molecule Scale (SNR)"); plt.ylabel("Bit Error Rate (BER)")
plt.legend(); plt.show()
