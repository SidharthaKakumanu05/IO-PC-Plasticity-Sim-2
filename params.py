import numpy as np

# Simulation
T_MAX = 300.0       # seconds
DT = 0.001          # time step (s)
SEED = 42

# Neuron counts
N_IO = 20
N_PF = 50
N_PC = 20

# IO frequencies (Hz)
IO_FREQS = [0.5, 1.0, 2.0]

# CF/PC mapping
CFS_PER_PC = 1   # tweakable: 1 = biological, N_IO = all-to-all

# Plasticity parameters
W_INIT = 0.5
ETA_LTP = 0.001
ETA_LTD = 0.009
LTP_WINDOW = 0.02   # 20 ms
LTD_WINDOW = 0.05   # 50 ms

# PF firing rate (Hz)
PF_RATE = 5.0  # realistic range ~1–10 Hz

rng = np.random.default_rng(SEED)
