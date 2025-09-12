# main_toy.py
import numpy as np
import argparse
import params
from neuron_models import PoissonNeuron
from plasticity import PurkinjeCell
from visualize_toy import plot_raster, plot_avg_weights, plot_individual_weights

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=float, default=params.T_MAX,
                        help="Simulation time in seconds (default from params.py)")
    parser.add_argument("--n_io", type=int, default=params.N_IO,
                        help="Number of IO neurons")
    parser.add_argument("--n_pc", type=int, default=params.N_PC,
                        help="Number of Purkinje cells")
    parser.add_argument("--n_pf", type=int, default=params.N_PF,
                        help="Number of PF inputs per PC")
    parser.add_argument("--cfs_per_pc", type=int, default=params.CFS_PER_PC,
                        help="Number of climbing fibers assigned to each PC")

    args = parser.parse_args()

    # overrides
    T = args.T
    n_io = args.n_io
    n_pc = args.n_pc
    n_pf = args.n_pf
    cfs_per_pc = args.cfs_per_pc

    dt = params.DT
    rng = np.random.default_rng(params.SEED)

    # Print active parameters for clarity
    print("="*50)
    print(f" Running Simulation")
    print(f" Duration:         {T} s")
    print(f" Time step:        {dt} s")
    print(f" IO neurons:       {n_io}")
    print(f" PCs:              {n_pc}")
    print(f" PFs per PC:       {n_pf}")
    print(f" CFs per PC:       {cfs_per_pc}")
    print(f" IO frequencies:   {params.IO_FREQS}")
    print("="*50)

    # Build IO neurons for each frequency condition
    io_spike_sets = []
    for f in params.IO_FREQS:
        io_neurons = [PoissonNeuron(f, T, rng) for _ in range(n_io)]
        io_spike_sets.append([n.spike_times for n in io_neurons])

    # Build PF neurons
    pf_neurons = [PoissonNeuron(params.PF_RATE, T, rng) for _ in range(n_pf)]
    pf_spike_times = [n.spike_times for n in pf_neurons]

    # CF mapping (random assignment)
    pc_cf_map = [
        rng.choice(n_io, size=cfs_per_pc, replace=(cfs_per_pc > n_io))
        for _ in range(n_pc)
    ]

    # Run for each IO frequency condition
    for f, io_spikes in zip(params.IO_FREQS, io_spike_sets):
        pcs = [PurkinjeCell(n_pf) for _ in range(n_pc)]

        # CF spikes per PC
        cf_spike_times_by_pc = [
            np.sort(np.concatenate([io_spikes[idx] for idx in cf_indices]))
            for cf_indices in pc_cf_map
        ]

        # Update PF→PC weights
        for pc_idx, pc in enumerate(pcs):
            cf_times = cf_spike_times_by_pc[pc_idx]
            for t_idx, t in enumerate(pf_spike_times[0]):  # PF event clocks
                pc.update_weights([t for _ in range(n_pf)], cf_times, t)

        # Save plots for the first PC (as representative)
        plot_raster(io_spikes, pf_spike_times, f)
        plot_avg_weights(pcs[0], cf_spike_times_by_pc[0], f)
        plot_individual_weights(pcs[0], cf_spike_times_by_pc[0], f)

if __name__ == "__main__":
    main()