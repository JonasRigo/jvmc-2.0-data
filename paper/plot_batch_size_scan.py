"""Speedup of the stages of a TDVP step (a) and peak memory (b) for a growing batch size, RWKV on 1 GPU.
Data from Scripts/batch_size_and_mc_chains_scan.py."""
import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "batch_size_scan" / "RWKV_L=100_Ns=32768_gpus=1.hdf5"

cfg.setup()

with h5py.File(DATA, "r") as f:
    # One group bc=<batch size> per point, sorted by batch size
    points = sorted(f.values(), key=lambda g: int(g.name.split("=")[1]))
    batch_size = np.array([int(g.name.split("=")[1]) for g in points])
    # Time of every stage in the last of the three steps
    times = {stage: np.array([g["timings"][stage]["last_timing"][()] for g in points]) for stage in cfg.STAGES}
    peak_memory = np.array([max(d["peak_bytes_in_use"][()] for d in g["memory"].values()) for g in points]) / 1e9

fig, (ax_speedup, ax_mem) = plt.subplots(1, 2, figsize=(13, 5))

# Speedup over the smallest batch size
for (stage, (label, color)), marker in zip(cfg.STAGES.items(), "osD"):
    ax_speedup.plot(batch_size, times[stage][0] / times[stage], marker=marker, color=color, label=label)
ax_mem.plot(batch_size, peak_memory, marker="o", color="k")

ax_speedup.set_ylabel(f"Speedup over batch size {batch_size[0]}")
ax_speedup.set_yscale("log")
cfg.plain_log_labels(ax_speedup.yaxis)
ax_speedup.set_title("(a)", loc="left")
ax_mem.set_ylabel("Peak device memory [GB]")
ax_mem.set_ylim(bottom=0)
ax_mem.set_title("(b)", loc="left")
for ax in (ax_speedup, ax_mem):
    ax.set_xscale("log", base=2)
    ax.set_xticks(2.0 ** np.arange(3, 16, 2))
    ax.set_xlabel("Batch size")

fig.legend(*ax_speedup.get_legend_handles_labels(), loc="upper center", ncol=len(cfg.STAGES), frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
cfg.save(fig, "batch_size_scan")
