import numpy as np

# -----------------------------
# Toy IO Neuron (forced oscillator with phase offset)
# -----------------------------
class IONeuronToy:
    def __init__(self, freq=1.0, dt=1e-3, phase=0.0):
        self.freq = freq   # in Hz
        self.dt = dt
        self.phase = phase # phase offset
        self.spikes = []

    def run(self, T=10.0):
        time = np.arange(0, T, self.dt)
        self.spikes = time[np.sin(2*np.pi*self.freq*time + self.phase) > 0.99]
        return self.spikes


# -----------------------------
# Purkinje Cell with toy plasticity
# -----------------------------
class PurkinjeCellToy:
    def __init__(self, w_init=0.5):
        self.w = w_init
        self.history = []
        self.spikes = []

    def apply_plasticity(self, pf_spike, cf_spikes,
                         ltd_window=0.1, null_window=0.2,
                         eta_ltd=0.01, eta_ltp=0.005):
        if len(cf_spikes) == 0:
            return
        delta_t = min([abs(pf_spike - cf) for cf in cf_spikes])
        if delta_t <= ltd_window:
            self.w -= eta_ltd * self.w
        elif delta_t > null_window:
            self.w += eta_ltp * (1 - self.w)
        self.w = np.clip(self.w, 0.0, 1.0)
        self.history.append(self.w)

    def receive_input(self, pf_spikes, cf_spikes, threshold=0.7):
        for pf in pf_spikes:
            if np.any(abs(cf_spikes - pf) < 0.05):
                if self.w > threshold:
                    self.spikes.append(pf)