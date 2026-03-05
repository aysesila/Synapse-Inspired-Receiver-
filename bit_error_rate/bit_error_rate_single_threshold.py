import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETTINGS (Fiziksel Kanal ve Haberleşme Parametreleri) ---
dt, T_bit = 0.005, 0.1
k_on, k_off = 50.0, 5.0
NUM_BITS = 2000
TRAIN_BITS = 200  # İlk 200 bit eşik (threshold) belirlemek için kullanılacak
SCALES = [5, 10, 20, 40, 80, 160]

# --- 2. MODELS ---
def simulate_standard(C):
    R, F = np.zeros(len(C)), np.ones(len(C))
    for i in range(1, len(C)):
        dR = (k_on * C[i-1] * F[i-1] - k_off * R[i-1]) * dt
        R[i] = R[i-1] + dR
        F[i] = max(0, 1.0 - R[i])
    return R

def simulate_tm(C, k_rec_val, is_dynamic=False, alpha=5.0): # Güçlü adaptasyon
    R, D, F = np.zeros(len(C)), np.zeros(len(C)), np.ones(len(C))
    for i in range(1, len(C)):
        # Dinamik modelin kalbi: Negatif Geri Besleme
        k_val = (0.2 + alpha * R[i-1]) if is_dynamic else k_rec_val
        dR = (k_on * C[i-1] * F[i-1] - k_off * R[i-1]) * dt
        dD = (k_off * R[i-1] - k_val * D[i-1]) * dt
        R[i], D[i] = R[i-1] + dR, D[i-1] + dD
        F[i] = max(0, 1.0 - R[i] - D[i])
    return R

# --- 3. SIGNAL (ISI Birikimi Yapan Difüzyon Kanalı) ---
np.random.seed(42)
bits = np.random.randint(0, 2, NUM_BITS)
time = np.arange(0, NUM_BITS * T_bit, dt)
clean_c = np.zeros(len(time))

for i, b in enumerate(bits):
    if b == 1:
        start_idx = int((i * T_bit) / dt)
        t_array = time[start_idx:] - time[start_idx]
        # Moleküller kanalda birikerek taban seviyesini (baseline) yükseltir
        clean_c[start_idx:] += 1.0 * np.exp(-t_array / 0.15) 

def get_responses(R_sig):
    res = []
    for i in range(NUM_BITS):
        idx = int((i * T_bit + 0.08) / dt) 
        res.append(R_sig[idx])
    return np.array(res)

# --- 4. EXECUTION (Preamble Thresholding) ---
# Sözlük anahtarlarını daha okunaklı ve raporlamaya uygun hale getirdik
res_map = { "Standard": [], "k_rec=0.2": [], "k_rec=0.45": [], "Dynamic": [] }

for s in SCALES:
    noisy = np.random.poisson(clean_c * s) / s
    outs = { "Standard": simulate_standard(noisy), 
             "k_rec=0.2": simulate_tm(noisy, 0.2),
             "k_rec=0.45": simulate_tm(noisy, 0.45), 
             "Dynamic": simulate_tm(noisy, 0, is_dynamic=True) }
    
    for key, R_sig in outs.items():
        resp = get_responses(R_sig)
        
        # SADECE EĞİTİM BİTLERİ (İlk 200 bit) ile eşik hesapla
        train_resp = resp[:TRAIN_BITS]
        train_bits = bits[:TRAIN_BITS]
        th = (np.mean(train_resp[train_bits==1]) + np.mean(train_resp[train_bits==0])) / 2
        
        # GERİ KALAN BİTLERİ (Data) bu sabit eşik ile test et
        test_resp = resp[TRAIN_BITS:]
        test_bits = bits[TRAIN_BITS:]
        
        ber = np.mean((test_resp > th).astype(int) != test_bits)
        res_map[key].append(float(ber))

# --- 5. DATA OUTPUT ---
print("\nSIMULATION RESULTS (PREAMBLE THRESHOLDING - REALISTIC)")
for model, values in res_map.items():
    print(f"{model}_BER = {values}")

# --- 6. PLOT ---
plt.figure(figsize=(9, 6))
styles = {"Standard": 'k*--', "k_rec=0.2": 'bo:', "k_rec=0.45": 'g^-.', "Dynamic": 'rs-'}
for k in ["Standard", "k_rec=0.2", "k_rec=0.45", "Dynamic"]:
    plt.plot(SCALES, [max(v, 1e-4) for v in res_map[k]], styles[k], label=k, linewidth=2)
plt.yscale('log'); plt.xscale('log'); plt.legend(); plt.grid(True, alpha=0.3)
plt.title("Realistic BER Performance (Training Sequence Method)", fontweight="bold")
plt.xlabel("Molecule Scale (SNR)"); plt.ylabel("Bit Error Rate (BER)")
plt.show()
