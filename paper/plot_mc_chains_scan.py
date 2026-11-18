"""Speedup of the sampling for a growing number of Markov chains, CpxCNN on 1, 2 and 4 GPUs.
Data from Scripts/batch_size_and_mc_chains_scan.py."""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

import plot_config as cfg

DATA = cfg.DATA_DIR / "mc_chains_scan"
SERIES = {
    1: ("1 GPU", cfg.COLORS["gpus_1"], "o"),
    2: ("2 GPUs", cfg.COLORS["gpus_2"], "s"),
    4: ("4 GPUs", cfg.COLORS["gpus_4"], "^"),
}

cfg.setup()

fig, ax = plt.subplots(figsize=cfg.figure_size(width_fraction=0.5, height_mm=75))

for n_gpus, (label, color, marker) in SERIES.items():
    with h5py.File(DATA / f"CpxCNN_L=50_Ns=4096_bs=4096_gpus={n_gpus}.hdf5", "r") as f:
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

fig.tight_layout()
cfg.save(fig, "mc_chains_scan")
