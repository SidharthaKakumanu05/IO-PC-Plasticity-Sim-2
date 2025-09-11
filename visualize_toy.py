import os
import numpy as np
import matplotlib.pyplot as plt

def plot_raster(io_spikes, pf_trains, T, freq, save_dir=None):
    plt.figure(figsize=(12,8))
    # IO (red) at rows 0..0 (just one IO train here)
    plt.scatter(io_spikes, [0]*len(io_spikes), color="red", marker="s", s=25, label="IO (CF)")

    # PFs (blue) stacked from row 5 upward
    offset = 5
    for i, spikes in enumerate(pf_trains):
        if len(spikes):
            plt.scatter(spikes, [offset+i]*len(spikes), color="blue", marker="|", s=15)

    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (grouped by type)")
    plt.title(f"Raster Plot (IO freq={freq} Hz)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/raster_{freq:.1f}Hz.png")
        plt.close()
    else:
        plt.show()


def plot_weights(pc, freq, save_dir=None):
    # Average weight over PFs as a function of PF-event time
    plt.figure(figsize=(9,4))
    if len(pc.avg_history_times):
        plt.plot(pc.avg_history_times, pc.avg_history_vals, color="purple")
    plt.xlabel("Time (s) (PF event times)")
    plt.ylabel("Average Synaptic Weight")
    plt.title("Average PF→PC Weight Evolution")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/weights_avg_{freq:.1f}Hz.png")
        plt.close()
    else:
        plt.show()

    # Individual PF trajectories (each only updates when that PF spikes)
    plt.figure(figsize=(12,5))
    for times, vals in zip(pc.indiv_times, pc.indiv_histories):
        if len(times):
            plt.plot(times, vals, alpha=0.5)
    plt.xlabel("Time (s) (PF event times)")
    plt.ylabel("Synaptic Weight")
    plt.title("PF→PC Weight Evolution (each PF)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/weights_individual_{freq:.1f}Hz.png")
        plt.close()
    else:
        plt.show()