import numpy as np
from neurons_toy import IONeuron, PFNeuron, PCNeuron

def run_simulation(io_freq=1.0, n_pfs=50, T=10.0, dt=0.001,
                   pf_rate=1.0, init_w=0.5,
                   eta_ltd=0.009, eta_ltp=0.001, ltd_window=0.05):
    """
    Simulate once; apply plasticity exactly once per PF spike in time order.
    Returns:
      io_spikes, pf_spike_trains (list of arrays), pc (with histories filled),
      pf_event_times (all PF spike times in chronological order).
    """
    # build neurons
    io = IONeuron(freq=io_freq)
    pfs = [PFNeuron(rate=pf_rate) for _ in range(n_pfs)]
    pc  = PCNeuron(n_pfs=n_pfs, init_weight=init_w)

    # generate spikes
    io_spikes = io.generate_spikes(T, dt)
    pf_trains = [pf.generate_spikes(T, dt) for pf in pfs]

    # merge ALL PF events into chronological stream of (time, pf_idx)
    events = []
    for idx, train in enumerate(pf_trains):
        if len(train):
            events.extend([(float(t), idx) for t in train])
    events.sort(key=lambda x: x[0])

    pf_event_times = []

    # walk events in time; update exactly once per PF spike
    for t_pf, idx in events:
        pc.plasticity_event(idx, t_pf, io_spikes,
                            eta_ltp=eta_ltp, eta_ltd=eta_ltd, ltd_window=ltd_window)
        pc.record_avg(t_pf)
        pf_event_times.append(t_pf)

    return io_spikes, pf_trains, pc, np.array(pf_event_times)