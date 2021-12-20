# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

ruta = ".\data\sim_from_2_to_10_sepqubits-entbases_from_3_to_5_bases.csv"
data = np.loadtxt(ruta, delimiter=",")
data = data.reshape(9, 3, 100)
media = np.mean(data, axis=2)
mediana = np.quantile(data, 0.5, axis=2)
cuantil_25 = np.quantile(data, 0.25, axis=2)
cuantil_75 = np.quantile(data, 0.75, axis=2)
qubits = np.array(["2", "3", "4", "5", "6", "7", "8", "9", "10"])
color = [np.array([0.500, 0.497, 0.791]),
         np.array([0.979, 0.744, 0.175]),
         np.array([0.500, 0.050, 0.500])]


def plots(qubits, media, mediana, cuantil_25, cuantil_75, color):
    fig, ax = plt.subplots(1,1, figsize=(6, 4))
    for k in range(3):
        ax.plot(qubits, cuantil_25[:, k], color=color[k]-0.05,
                linewidth=0.3)
        ax.plot(qubits, mediana[:, k], "--o", color=color[k],
                markersize=7)
        ax.plot(qubits, cuantil_75[:, k], color=color[k]-0.05,
                linewidth=0.3)
        ax.fill_between(qubits, cuantil_25[:, k], cuantil_75[:, k],
                        where=cuantil_75[:, k] >= cuantil_25[:, k],
                        facecolor=color[k]-0.05, interpolate=True,
                        alpha=0.2)
        ax.set_ylabel('Fidelity', size=20)
        ax.set_xlabel('Number of qubits', size=20)
        ax.set_ylim((0.0, 1.05))
        ax.grid()
    rcParams.update({'font.size': 20})
    # plt.yticks([0.85, 0.9, 0.95, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

x = 5
fig = plots(qubits[:x], media[:x, :], mediana[:x, :],
            cuantil_25[:x, :], cuantil_75[:x, :], color)
plt.savefig(".\otros\FIGURE2-4.pdf", bbox_inches='tight')
