from simulation_toy import run_toy
from visualize_toy import plot_toy_results

if __name__ == "__main__":
    # Run demo at three IO frequencies
    for freq in [0.5, 1.0, 2.0]:
        time, cf_spikes, pf_spikes, pc_spikes, weights = run_toy(freq=freq, pf_rate=5.0, T=10.0)
        plot_toy_results(time, cf_spikes, pf_spikes, pc_spikes, weights, freq)