import matplotlib.pyplot as plt

def plot_toy_results(time, cf_spikes, pf_spikes, pc_spikes, weights, freq):
    # Raster plot
    plt.figure(figsize=(10,6))
    plt.scatter(cf_spikes, [2]*len(cf_spikes), color="red", marker="|", label="IO (CF)")
    plt.scatter(pf_spikes, [1]*len(pf_spikes), color="blue", marker="|", label="PF")
    plt.scatter(pc_spikes, [0]*len(pc_spikes), color="green", marker="|", label="PC")
    plt.yticks([0,1,2], ["PC", "PF", "IO"])
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron")
    plt.title(f"Spike Raster (IO freq={freq} Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Weight evolution
    plt.figure(figsize=(7,4))
    plt.plot(weights, color="purple")
    plt.xlabel("Update step")
    plt.ylabel("Synaptic Weight")
    plt.title("PF→PC Weight Evolution")
    plt.tight_layout()
    plt.show()