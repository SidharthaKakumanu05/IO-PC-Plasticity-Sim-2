import numpy as np
from neurons_toy import IONeuronToy, PurkinjeCellToy

# -----------------------------
# PF input (Poisson process)
# -----------------------------
def generate_pf_spikes(rate=5.0, T=10.0, dt=1e-3, n_pf=50):
    time = np.arange(0, T, dt)
    pf_trains = []
    for _ in range(n_pf):
        spikes = time[np.random.rand(len(time)) < rate*dt]
        pf_trains.append(spikes)
    return pf_trains


# -----------------------------
# Run sparse toy IO–PF–PC network
# -----------------------------
def run_toy_network(freq=1.0, pf_rate=5.0, T=10.0, dt=1e-3,
                    n_io=20, n_pf=50, n_pc=10, pf_per_pc_range=(5,15)):
    time = np.arange(0, T, dt)

    # IO neurons
    io_neurons = [IONeuronToy(freq=freq, dt=dt, phase=np.random.rand()*2*np.pi)
                  for _ in range(n_io)]
    io_spike_trains = [io.run(T=T) for io in io_neurons]

    # PF neurons
    pf_trains = generate_pf_spikes(rate=pf_rate, T=T, dt=dt, n_pf=n_pf)

    # PC neurons
    pcs = []
    pc_spike_trains = []
    all_weights = []

    for _ in range(n_pc):
        pc = PurkinjeCellToy()
        # Random PF subset size
        n_conn = np.random.randint(pf_per_pc_range[0], pf_per_pc_range[1]+1)
        pf_indices = np.random.choice(range(n_pf), n_conn, replace=False)
        # Apply plasticity for chosen PFs
        for idx in pf_indices:
            for pf in pf_trains[idx]:
                pc.apply_plasticity(pf, np.concatenate(io_spike_trains))
        # Generate PC spikes
        for idx in pf_indices:
            pc.receive_input(pf_trains[idx], np.concatenate(io_spike_trains))
        pcs.append(pc)
        pc_spike_trains.append(pc.spikes)
        all_weights.append(pc.history)

    return time, io_spike_trains, pf_trains, pc_spike_trains, all_weights