"""Quench dynamics of the 2D transverse-field Ising model at h = h_c, starting in the x-polarized state, for L x L
lattices. Rows: energy, TDVP error and magnetization X; columns: L = 6, 8, 10.

Data in Data/tfim_dynamics/, from the runs in Scripts/tfim_dynamics/ (encap version names below) and the references:
    jvmc_6x6_nc10.csv      6_by_6       ResNet, n_C = 10, 8e4 samples
    jvmc_8x8_nc16.csv      Ns_3e5       ResNet, n_C = 16, 3e5 samples
    jvmc_8x8_nc20.csv      larger_net   ResNet, n_C = 20, 1e5 samples
    jvmc_10x10_nc16.csv    10_by_10     ResNet, n_C = 16, 3e5 samples
    exact_6x6.csv          exact X (columns x = h_c t, y = X)
    cnn_chen_8x8.npy       CNN of Chen et al., X at h_c t = 1e-3 h_c (1, 2, ...)
    cnn_schmitt_8x8.csv    CNN of Schmitt et al. (columns t, Sx)
    cnn_schmitt_10x10.csv  CNN of Schmitt et al. (columns x = h_c t, y = X)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "tfim_dynamics"
H_C = 3.04438
# Larger than cfg.FONTSIZE, for a 3x3 grid that is shrunk to the page width
FONTSIZE = 16
# jVMC runs per lattice size: (file, label, color)
RUNS = {
    6: [("jvmc_6x6_nc10.csv", r"jVMC-2.0, $n_C=10$", cfg.COLORS["tdvp"])],
    8: [("jvmc_8x8_nc16.csv", r"jVMC-2.0, $n_C=16$", cfg.COLORS["tdvp"]),
        ("jvmc_8x8_nc20.csv", r"jVMC-2.0, $n_C=20$", cfg.COLORS["tdvp_large"])],
    10: [("jvmc_10x10_nc16.csv", r"jVMC-2.0, $n_C=16$", cfg.COLORS["tdvp"])],
}
LABEL_CHEN = "CNN (Chen et al.)"
LABEL_SCHMITT = "CNN (Schmitt et al.)"

def load_run(file):
    # Columns by header name. The 6x6 header has one name more than there are columns (a column after tdvp_error was
    # not written), so there only the columns up to tdvp_error, all that is plotted here, match their names
    path = DATA / file
    names = path.open().readline().split()
    values = np.loadtxt(path, skiprows=1, unpack=True)
    return dict(zip(names, values))

cfg.setup()
mpl.rcParams.update({key: FONTSIZE for key in
                     ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize")})

fig, axes = plt.subplots(3, len(RUNS), figsize=(12, 7.5), sharex=True, sharey="row")
for col, (L, runs) in zip(axes.T, RUNS.items()):
    ax_E, ax_r, ax_X = col
    for file, label, color in runs:
        data = load_run(file)
        t = data["time"] * H_C
        # Energy relative to the initial state, E_0 = -h_c L^2
        ax_E.plot(t, data["energy_mean"] + H_C * L**2, color=color)
        ax_r.plot(t, data["tdvp_error"], color=color)
        ax_X.plot(t, data["X_mean"], color=color, label=label)
    ax_E.set_title(rf"${L}\times {L}$")

ax_X = axes[2, 0]
x, X = np.loadtxt(DATA / "exact_6x6.csv", skiprows=1, unpack=True)
ax_X.plot(x, X, "o", color=cfg.COLORS["exact"], label="exact")

ax_X = axes[2, 1]
X = np.load(DATA / "cnn_chen_8x8.npy")
ax_X.plot(1e-3 * H_C * np.arange(1, len(X) + 1), X, color=cfg.COLORS["cnn_chen"], label=LABEL_CHEN)
data = np.genfromtxt(DATA / "cnn_schmitt_8x8.csv", delimiter=",", names=True)
ax_X.plot(H_C * data["t"], data["Sx"], color=cfg.COLORS["cnn_schmitt"], label=LABEL_SCHMITT)

ax_X = axes[2, 2]
x, X = np.loadtxt(DATA / "cnn_schmitt_10x10.csv", delimiter=",", skiprows=1, unpack=True)
ax_X.plot(x, X, color=cfg.COLORS["cnn_schmitt"], label=LABEL_SCHMITT)

for ax in axes[1]:
    ax.set_yscale("log")
for ax in axes[2]:
    ax.set_xlabel(r"time $h_\mathrm{c} t$")
    ax.legend(loc="upper right", fontsize=FONTSIZE - 3, labelspacing=0.3, borderaxespad=0.3)

axes[0, 0].set_ylabel(r"energy $E - E_0$")
axes[1, 0].set_ylabel(r"TDVP error $r^2$")
axes[2, 0].set_ylabel(r"magnetization $X$")
axes[0, 0].set(xlim=(0, 6.1), xticks=range(7), ylim=(-2, 2), yticks=[-2, 0, 2])
axes[1, 0].set(ylim=(1e-6, 0.2), yticks=[1e-6, 1e-4, 1e-2])
axes[2, 0].set(ylim=(0.75, 1), yticks=[0.8, 0.9, 1.0])

fig.tight_layout()
cfg.save(fig, "tfim_dynamics")
