from simulation_toy import run_simulation
from visualize_toy import plot_raster, plot_weights

def main():
    T = 10.0
    dt = 0.001
    n_pfs = 50
    pf_rate = 1.0

    for f in [0.5, 1.0, 2.0]:
        io_spikes, pf_trains, pc = run_simulation(
            io_freq=f, n_pfs=n_pfs, T=T, dt=dt,
            pf_rate=pf_rate,
            eta_ltd=0.009, eta_ltp=0.001,  # 9:1 ratio
            ltd_window=0.05                # 50 ms LTD window
        )
        plot_raster(io_spikes, pf_trains, T, f, save_dir="output")
        plot_weights(pc, f, save_dir="output")

if __name__ == "__main__":
    main()