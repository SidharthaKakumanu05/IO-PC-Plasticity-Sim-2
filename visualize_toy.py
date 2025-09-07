import matplotlib.pyplot as plt
import numpy as np

def plot_raster(io_spike_trains, pf_trains, pc_spike_trains, T, freq):
    plt.figure(figsize=(12,8))

    # IOs (red, top block)
    offset = 0
    for i, spikes in enumerate(io_spike_trains):
        plt.scatter(spikes, [offset+i]*len(spikes), color="red", marker="|", s=20)
    offset += len(io_spike_trains)

    # PFs (blue, middle block)
    for i, spikes in enumerate(pf_trains):
        plt.scatter(spikes, [offset+i]*len(spikes), color="blue", marker="|", s=20)
    offset += len(pf_trains)

    # PCs (green, bottom block)
    for i, spikes in enumerate(pc_spike_trains):
        plt.scatter(spikes, [offset+i]*len(spikes), color="green", marker="|", s=30)

    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (grouped by type)")
    plt.title(f"Raster Plot (IO freq={freq} Hz)")
    plt.tight_layout()
    plt.show()


def plot_weights(all_weights):
    plt.figure(figsize=(10,5))

    # Individual PC weight traces
    for w in all_weights:
        if len(w) > 0:
            plt.plot(w, alpha=0.6)

    plt.xlabel("Update step")
    plt.ylabel("Synaptic Weight")
    plt.title("PF→PC Weight Evolution (all PCs)")
    plt.tight_layout()
    plt.show()

    # Average trace
    min_len = min(len(w) for w in all_weights if len(w) > 0)
    if min_len > 0:
        avg_trace = np.mean([w[:min_len] for w in all_weights if len(w) > 0], axis=0)
        plt.figure(figsize=(7,4))
        plt.plot(avg_trace, color="purple")
        plt.xlabel("Update step")
        plt.ylabel("Average Synaptic Weight")
        plt.title("Average PF→PC Weight Evolution")
        plt.tight_layout()
        plt.show()