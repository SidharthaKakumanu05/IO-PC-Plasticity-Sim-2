import matplotlib.pyplot as plt
import numpy as np
import os

def plot_raster(io_spikes, pf_spikes, freq, save_dir="outputs"):
    plt.figure(figsize=(12, 5))
    for i, st in enumerate(io_spikes):
        plt.vlines(st, i - 0.4, i + 0.4, color='r')
    offset = len(io_spikes)
    for j, st in enumerate(pf_spikes):
        plt.vlines(st, offset + j - 0.4, offset + j + 0.4, color='b')

    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index (IOs red, PFs blue)")
    plt.title(f"Raster Plot (IO freq={freq} Hz)")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"raster_{freq}Hz.png"))
    plt.close()

def plot_avg_weights(pc, cf_times, freq, save_dir="outputs"):
    times = np.arange(len(pc.avg_history))
    plt.figure(figsize=(8, 4))
    plt.plot(times, pc.avg_history, color='purple', label="Average weight")

    # Shade every 20th CF window for readability
    for k, tcf in enumerate(cf_times):
        if k % 20 == 0:
            plt.axvspan(tcf - 0.025, tcf + 0.025, color='k', alpha=0.05)

    plt.xlabel("Time (s)")
    plt.ylabel("Average Synaptic Weight")
    plt.title(f"Average PF→PC Weight (IO {freq} Hz)")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"weights_avg_{freq}Hz.png"))
    plt.close()

def plot_individual_weights(pc, cf_times, freq, save_dir="outputs", n_sample=10):
    sampled = np.random.choice(len(pc.indiv_histories), size=min(n_sample, len(pc.indiv_histories)), replace=False)
    plt.figure(figsize=(12, 5))
    for idx in sampled:
        plt.plot(pc.indiv_histories[idx], alpha=0.7)

    plt.plot(pc.avg_history, color='purple', linewidth=2, label="Average weight")
    plt.xlabel("Time (s)")
    plt.ylabel("Synaptic weight")
    plt.title(f"PF→PC Weight Evolution (IO {freq} Hz, {len(sampled)} sampled PFs)")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"weights_individual_{freq}Hz.png"))
    plt.close()