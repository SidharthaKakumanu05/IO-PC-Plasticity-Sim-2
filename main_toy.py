from simulation_toy import run_simulation
from visualize_toy import plot_raster, plot_weights

def main():
    T = 10.0
    dt = 0.001
    n_pfs = 50
    pf_rate = 1.0

    for f in [0.5, 1.0, 2.0]:
        io_spikes, pf_trains, pc, pf_event_times = run_simulation(
            io_freq=f, n_pfs=n_pfs, T=T, dt=dt,
            pf_rate=pf_rate,
            eta_ltd=0.009, eta_ltp=0.001,  # 9:1 ratio
            ltd_window=0.05                # 50 ms LTD window
        )

        # Raster plot
        plot_raster(io_spikes, pf_trains, T, f, save_dir="outputs")

        # Weight evolution plots (avg + individual)
        plot_weights(pc, pf_event_times, f, save_dir="outputs")

if __name__ == "__main__":
    main()