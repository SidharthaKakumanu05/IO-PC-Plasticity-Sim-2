import numpy as np
from neurons_toy import IONeuronToy, PurkinjeCellToy

# -----------------------------
# PF input (Poisson process)
# -----------------------------
def generate_pf_spikes(rate=5.0, T=10.0, dt=1e-3):
    time = np.arange(0, T, dt)
    pf_spikes = time[np.random.rand(len(time)) < rate*dt]
    return pf_spikes


# -----------------------------
# Run toy IO–PF–PC network
# -----------------------------
def run_toy(freq=1.0, pf_rate=5.0, T=10.0, dt=1e-3):
    time = np.arange(0, T, dt)
    
    # IO
    io = IONeuronToy(freq=freq, dt=dt)
    cf_spikes = io.run(T=T)
    
    # PF
    pf_spikes = generate_pf_spikes(rate=pf_rate, T=T, dt=dt)
    
    # PC
    pc = PurkinjeCellToy()
    for pf in pf_spikes:
        pc.apply_plasticity(pf, cf_spikes)
    pc.receive_input(pf_spikes, cf_spikes)

    return time, cf_spikes, pf_spikes, pc.spikes, pc.history