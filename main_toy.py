from simulation_toy import run_toy_network
from visualize_toy import plot_raster, plot_weights

if __name__ == "__main__":
    # Run network at different IO frequencies
    for freq in [0.5, 1.0, 2.0]:
        time, io_spike_trains, pf_trains, pc_spike_trains, all_weights = run_toy_network(
            freq=freq, pf_rate=5.0, T=10.0,
            n_io=20, n_pf=50, n_pc=10, pf_per_pc_range=(5,15)
        )
        plot_raster(io_spike_trains, pf_trains, pc_spike_trains, T=10.0, freq=freq)
        plot_weights(all_weights)