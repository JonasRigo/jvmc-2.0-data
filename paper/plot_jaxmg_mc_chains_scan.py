"""Speedup of the sampling for a growing number of Markov chains, CpxCNN on 1, 2 and 4 GPUs (a), and time of a TDVP
step with the SR equation solved on one device or distributed with jaxmg, RNN on 16 GPUs (b).
Data from Scripts/batch_size_and_mc_chains_scan.py and Scripts/jaxmg_and_gpu_scaling.py."""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

import plot_config as cfg

JAXMG_DATA = cfg.DATA_DIR / "jaxmg_scan"
MC_CHAINS_DATA = cfg.DATA_DIR / "mc_chains_scan"
JAXMG_SERIES = {
    "off": ("jaxmg off, 16 GPUs", cfg.COLORS["jaxmg_off"], "o"),
    "on": ("jaxmg on, 16 GPUs", cfg.COLORS["jaxmg_on"], "s"),
}
MC_CHAINS_SERIES = {
    1: ("1 GPU", cfg.COLORS["gpus_1"], "o"),
    2: ("2 GPUs", cfg.COLORS["gpus_2"], "s"),
    4: ("4 GPUs", cfg.COLORS["gpus_4"], "^"),
}

def plot_jaxmg(ax):
    for jaxmg, (label, color, marker) in JAXMG_SERIES.items():
        points = []
        for path in JAXMG_DATA.glob(f"*_jaxmg={jaxmg}.hdf5"):
            with h5py.File(path, "r") as f:
                # Total time of the last of the three steps
                points.append((f["metadata/num_params"][()], sum(f["timings"][s]["last_timing"][()] for s in cfg.STAGES)))
        n_params, times = np.array(sorted(points)).T
        ax.plot(n_params, times, marker=marker, color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Time of one TDVP step [s]")
    cfg.plain_log_labels(ax.yaxis)
    ax.legend()

def plot_mc_chains(ax):
    for n_gpus, (label, color, marker) in MC_CHAINS_SERIES.items():
        with h5py.File(MC_CHAINS_DATA / f"CpxCNN_L=50_Ns=4096_bs=4096_gpus={n_gpus}.hdf5", "r") as f:
            # One group mc=<number of chains> per point, sorted by the number of chains
            points = sorted(f.values(), key=lambda g: int(g.name.split("=")[1]))
            chains = np.array([int(g.name.split("=")[1]) for g in points])
            # Time of the sampling in the last of the three steps
            times = np.array([g["timings"]["sampling"]["last_timing"][()] for g in points])
        # Speedup over the smallest number of chains on the same number of GPUs
        ax.plot(chains, times[0] / times, marker=marker, color=color, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Markov chains")
    ax.set_ylabel(f"Speedup of the sampling over {chains[0]} chains")
    cfg.plain_log_labels(ax.xaxis)
    # Minor labels would overlap over almost 3 decades
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    cfg.plain_log_labels(ax.yaxis)
    # Below the curves, which leave the lower middle empty
    ax.legend(loc="lower center", bbox_to_anchor=(0.6, 0))

cfg.setup()

fig, (ax_mc_chains, ax_jaxmg) = plt.subplots(1, 2, figsize=cfg.figure_size(height_mm=80))
plot_mc_chains(ax_mc_chains)
plot_jaxmg(ax_jaxmg)
ax_mc_chains.set_title("(a)", loc="left")
ax_jaxmg.set_title("(b)", loc="left")

fig.tight_layout()
cfg.save(fig, "jaxmg_mc_chains_scan")
