# Synapse-Inspired-Receiver-
Computation-Free ISI Mitigation for Molecular Communication
Diffusion-based molecular communication suffers from Inter-Symbol Interference (ISI) — molecules from previous transmissions linger in the channel and corrupt subsequent signals. 
Standard solutions rely on digital filters, which are impractical for power-constrained nanorobots.
This project proposes a bio-inspired receiver that mitigates ISI without any computation. By adapting the Tsodyks-Markram model of short-term synaptic depression, the receptor is given a three-state dynamic (Free → Bound → Depressed) that causes it to physically ignore residual molecules from previous bits. The recovery time constant τ_rec is tuned to match the channel's diffusion decay, effectively turning the receptor into a passive high-pass filter.
Simulations in Python demonstrate clear ISI suppression across multiple bit patterns (1010, 1100, 1001) and pulse shapes (exponential, rectangular, Gaussian), compared against a standard ligand-receptor baseline.

Dependencies: numpy scipy matplotlib
References:
Pierobon & Akan (2010) 
Abbott & Regehr (2004) 
Tsodyks & Markram (1997)
