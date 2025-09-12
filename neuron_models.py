import numpy as np

class PoissonNeuron:
    def __init__(self, rate_hz, T, rng, dt=0.001):
        self.rate = rate_hz
        self.T = T
        self.dt = dt
        self.rng = rng
        self.spike_times = self._generate_spikes()

    def _generate_spikes(self):
        n_steps = int(self.T / self.dt)
        p_spike = self.rate * self.dt
        spikes = self.rng.random(n_steps) < p_spike
        return np.where(spikes)[0] * self.dt