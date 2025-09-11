import numpy as np

class IONeuron:
    """Inferior Olive neuron (drives CF spikes into PCs)."""
    def __init__(self, freq=1.0):
        self.freq = freq

    def generate_spikes(self, T, dt):
        isi = 1.0 / max(self.freq, 1e-9)
        spikes = np.arange(0.0, T + 1e-12, isi)
        return spikes[(spikes >= 0.0) & (spikes < T)]


class PFNeuron:
    """Parallel Fiber neuron (background Poisson input)."""
    def __init__(self, rate=1.0):
        self.rate = rate

    def generate_spikes(self, T, dt):
        t = np.arange(0.0, T, dt)
        return t[np.random.rand(len(t)) < self.rate * dt]


class PCNeuron:
    """
    Purkinje Cell with PF->PC plasticity.
    We keep one weight per PF input.
    """
    def __init__(self, n_pfs, init_weight=0.5):
        self.weights = np.ones(n_pfs, dtype=float) * init_weight
        # histories
        self.avg_history_times = []
        self.avg_history_vals  = []
        self.indiv_histories   = [[] for _ in range(n_pfs)]  # per-PF trajectory (piecewise)
        self.indiv_times       = [[] for _ in range(n_pfs)]

    def plasticity_event(self, pf_idx, pf_time, cf_spikes,
                         eta_ltp=0.001, eta_ltd=0.009, ltd_window=0.05):
        """
        Apply a single plasticity update for a PF spike at time pf_time.
        LTD if any CF is within ltd_window; otherwise LTP.
        """
        if len(cf_spikes) and np.any(np.abs(cf_spikes - pf_time) <= ltd_window):
            # LTD
            self.weights[pf_idx] -= eta_ltd * self.weights[pf_idx]
        else:
            # LTP
            self.weights[pf_idx] += eta_ltp * (1.0 - self.weights[pf_idx])

        # clip
        self.weights[pf_idx] = float(np.clip(self.weights[pf_idx], 0.0, 1.0))

        # record per-PF trajectory point
        self.indiv_histories[pf_idx].append(self.weights[pf_idx])
        self.indiv_times[pf_idx].append(pf_time)

    def record_avg(self, t_now):
        self.avg_history_times.append(t_now)
        self.avg_history_vals.append(float(np.mean(self.weights)))