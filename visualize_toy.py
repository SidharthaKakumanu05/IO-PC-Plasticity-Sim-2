import matplotlib.pyplot as plt
import numpy as np

def plot_raster(io_spike_trains, pf_trains, pc_spike_trains, T, freq):
    plt.figure(figsize=(10,7))

    # IOs (red)
    for i, spikes in enumerate(io_spike_trains):
        plt.scatter(spikes, [2+i]*len(spikes), color="red", marker="|")
    # PFs (blue)
    for i, spikes in enumerate(pf_trains):
        plt.scatter(spikes, [100+i]*len(spikes), color="blue", marker="|")
    # PCs (green)
    for i, spikes in enumerate(pc_spike_trains):
        plt.scatter(spikes, [200+i]*len(spikes), color="green", marker="|")

    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (grouped by type)")
    plt.title(f"Raster Plot (IO freq={freq} Hz)")
    plt.tight_layout()
    plt.show()


def plot_weights(all_weights):
    plt.figure(figsize=(10,5))

    # Individual PC weight traces
    for w in all_weights:
        plt.plot(w, alpha=0.6)

    plt.xlabel("Update step")
    plt.ylabel("Synaptic Weight")
    plt.title("PF→PC Weight Evolution (all PCs)")
    plt.tight_layout()
    plt.show()

    # Average trace
    avg_len = min(len(w) for w in all_weights)
    avg_trace = np.mean([w[:avg_len] for w in all_weights], axis=0)

    plt.figure(figsize=(7,4))
    plt.plot(avg_trace, color="purple")
    plt.xlabel("Update step")
    plt.ylabel("Average Synaptic Weight")
    plt.title("Average PF→PC Weight Evolution")
    plt.tight_layout()
    plt.show()