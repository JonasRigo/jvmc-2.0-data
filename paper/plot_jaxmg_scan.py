"""Time of a TDVP step with the SR equation solved on one device or distributed with jaxmg, for a growing
number of parameters, RNN on 16 GPUs. Data from Scripts/jaxmg_and_gpu_scaling.py."""
import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "jaxmg_scan"
SERIES = {
    "off": ("jaxmg off, 16 GPUs", cfg.COLORS["jaxmg_off"], "o"),
    "on": ("jaxmg on, 16 GPUs", cfg.COLORS["jaxmg_on"], "s"),
}

cfg.setup()

fig, ax = plt.subplots(figsize=cfg.figure_size(width_fraction=0.5, height_mm=75))

for jaxmg, (label, color, marker) in SERIES.items():
    points = []
    for path in DATA.glob(f"*_jaxmg={jaxmg}.hdf5"):
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

fig.tight_layout()
cfg.save(fig, "jaxmg_scan")
