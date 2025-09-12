import numpy as np
import params
from neuron_models import PoissonNeuron
from plasticity import PurkinjeCell
from visualize_toy import plot_raster, plot_avg_weights, plot_individual_weights

def main():
    rng = np.random.default_rng(params.SEED)

    # Build IO neurons
    io_spike_sets = []
    for f in params.IO_FREQS:
        io_neurons = [PoissonNeuron(f, params.T_MAX, rng) for _ in range(params.N_IO)]
        io_spike_sets.append([n.spike_times for n in io_neurons])

    # Build PFs
    pf_neurons = [PoissonNeuron(params.PF_RATE, params.T_MAX, rng) for _ in range(params.N_PF)]
    pf_spike_times = [n.spike_times for n in pf_neurons]

    # CF mapping (per PC)
    pc_cf_map = [
        rng.choice(params.N_IO, size=params.CFS_PER_PC, replace=(params.CFS_PER_PC > params.N_IO))
        for _ in range(params.N_PC)
    ]

    # Run for each IO frequency condition
    for f, io_spikes in zip(params.IO_FREQS, io_spike_sets):
        pcs = [PurkinjeCell(params.N_PF) for _ in range(params.N_PC)]

        # CF spikes per PC
        cf_spike_times_by_pc = [
            np.sort(np.concatenate([io_spikes[idx] for idx in cf_indices]))
            for cf_indices in pc_cf_map
        ]

        # Update PF→PC weights over time
        for pc_idx, pc in enumerate(pcs):
            cf_times = cf_spike_times_by_pc[pc_idx]
            for t_idx, t in enumerate(pf_spike_times[0]):  # all PFs have similar clocks
                pc.update_weights([t for _ in range(params.N_PF)], cf_times, t)

        # Save outputs
        plot_raster(io_spikes, pf_spike_times, f)
        plot_avg_weights(pcs[0], cf_spike_times_by_pc[0], f)
        plot_individual_weights(pcs[0], cf_spike_times_by_pc[0], f)

if __name__ == "__main__":
    main()