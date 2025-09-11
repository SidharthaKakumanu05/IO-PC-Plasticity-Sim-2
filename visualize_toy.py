import os
import matplotlib.pyplot as plt

def plot_raster(io_spikes, pf_trains, T, freq, save_dir="outputs"):
    """Raster plot of IO and PF spikes."""
    plt.figure(figsize=(12,6))

    # IO (red, at row 0)
    plt.scatter(io_spikes, [0]*len(io_spikes), color="red", marker="s", s=25, label="IO (CF)")

    # PFs (blue, stacked from row 5 upward)
    offset = 5
    for i, spikes in enumerate(pf_trains):
        if len(spikes):
            plt.scatter(spikes, [offset+i]*len(spikes), color="blue", marker="|", s=15)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (grouped by type)")
    plt.title(f"Raster Plot (IO freq={freq} Hz)")
    plt.legend(loc="upper right")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"raster_{freq:.1f}Hz.png"))
    plt.close()


def plot_weights_avg(pc, pf_event_times, freq, save_dir="outputs"):
    """Average PF->PC weight evolution with PF spike markers."""
    plt.figure(figsize=(9,4))
    if len(pc.avg_history_times):
        plt.plot(pc.avg_history_times, pc.avg_history_vals, color="purple", label="Average weight")

    # PF spike markers
    for t in pf_event_times:
        plt.axvline(t, color="gray", linestyle="--", alpha=0.2)

    plt.xlabel("Time (s) (PF event times)")
    plt.ylabel("Average Synaptic Weight")
    plt.title("Average PF→PC Weight Evolution")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"weights_avg_{freq:.1f}Hz.png"))
    plt.close()


def plot_weights_individual(pc, pf_event_times, freq, save_dir="outputs"):
    """Individual PF->PC weight trajectories with PF spike markers."""
    plt.figure(figsize=(12,5))
    for times, vals in zip(pc.indiv_times, pc.indiv_histories):
        if len(times):
            plt.plot(times, vals, alpha=0.7)

    # PF spike markers
    for t in pf_event_times:
        plt.axvline(t, color="gray", linestyle="--", alpha=0.2)

    plt.xlabel("Time (s) (PF event times)")
    plt.ylabel("Synaptic Weight")
    plt.title("PF→PC Weight Evolution (each PF)")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"weights_individual_{freq:.1f}Hz.png"))
    plt.close()


def plot_weights(pc, pf_event_times, freq, save_dir="outputs"):
    """
    Backward-compatible wrapper: generate both average + individual plots.
    """
    plot_weights_avg(pc, pf_event_times, freq, save_dir)
    plot_weights_individual(pc, pf_event_times, freq, save_dir)