import numpy as np
import params

class PurkinjeCell:
    def __init__(self, n_pf):
        self.weights = np.full(n_pf, params.W_INIT)
        self.indiv_histories = [[] for _ in range(n_pf)]
        self.avg_history = []

    def update_weights(self, pf_spike_times, cf_spike_times, current_time):
        """Update weights given PF spikes and this PC's CF spikes."""
        for i, t_pf in enumerate(pf_spike_times):
            # Find nearest CF
            dt = np.min(np.abs(cf_spike_times - t_pf)) if len(cf_spike_times) else np.inf

            if dt < params.LTD_WINDOW:
                self.weights[i] -= params.ETA_LTD * self.weights[i]
            elif dt > params.LTP_WINDOW:
                self.weights[i] += params.ETA_LTP * (1.0 - self.weights[i])

            # clamp
            self.weights[i] = np.clip(self.weights[i], 0.0, 1.0)
            self.indiv_histories[i].append(self.weights[i])

        self.avg_history.append(np.mean(self.weights))