"""Speedup of a TDVP step with the number of GPUs (a) and time of its stages on 1 and 16 GPUs (b), RNN without
jaxmg. Data from Scripts/jaxmg_and_gpu_scaling.py."""
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np

import plot_config as cfg

DATA = cfg.DATA_DIR / "gpu_scaling"
# Numbers of GPUs with a breakdown in (b)
BREAKDOWN_GPUS = [1, 16]
# Stage labels broken over lines, so that they take less space next to the donuts
DONUT_LABELS = {"compute objective function and gradient": "Objective function\nand gradient"}

def spread(ys, min_sep):
    # Shift the heights of the labels apart, keeping their order and mean, so that neighbors are min_sep apart
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    out = ys.copy()
    for prev, cur in zip(order[:-1], order[1:]):
        out[cur] = max(out[cur], out[prev] + min_sep)
    return out + (ys - out).mean()

def plot_donut(ax, times, n_gpus):
    values = [times[s] for s in cfg.STAGES]
    total = sum(values)

    # The largest stage on the left, the small ones centered on the right, where their labels go
    frac = values[0] / total
    wedges, _ = ax.pie(values, colors=[color for _, color in cfg.STAGES.values()],
                       startangle=-360 * frac - 180 * (1 - frac), counterclock=True,
                       wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5))
    ax.text(0, 0.1, f"{n_gpus} GPU{'s' if n_gpus > 1 else ''}", ha="center", va="center", weight="bold")
    ax.text(0, -0.18, f"{total:.1f} s", ha="center", va="center", color=cfg.COLORS["guide"])

    # One label per stage, on the side of the middle of its wedge
    anchors = []
    for wedge, stage, value in zip(wedges, cfg.STAGES, values):
        label = DONUT_LABELS.get(stage, cfg.STAGES[stage][0])
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        # anchors.append((np.cos(angle), np.sin(angle), f"{label}\n{value:.2f} s ({100 * value / total:.1f}%)"))
        anchors.append((np.cos(angle), np.sin(angle), f"{label}\n{value:.2f} s"))
    for side in (-1, 1):
        idx = [i for i, (x, _, _) in enumerate(anchors) if (x < 0) == (side < 0)]
        for i, label_y in zip(idx, spread([1.2 * anchors[i][1] for i in idx], min_sep=0.6)):
            x, y, text = anchors[i]
            ax.annotate(text, xy=(0.98 * x, 0.98 * y), xytext=(1.35 * side, label_y), ha="left" if side > 0 else "right",
                        va="center", fontsize=cfg.TICK_FONTSIZE,
                        arrowprops=dict(arrowstyle="-", color=cfg.COLORS["guide"], linewidth=0.8,
                                        connectionstyle=f"arc,angleA={180 if side > 0 else 0},"
                                                        f"angleB={0 if side > 0 else 180},armA=0,armB=10,rad=0"))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

cfg.setup()

# Time of every stage in the last of the three steps, for every number of GPUs
times = {}
for path in DATA.glob("*.hdf5"):
    with h5py.File(path, "r") as f:
        times[int(re.search(r"gpus=(\d+)", path.name).group(1))] = {s: f["timings"][s]["last_timing"][()] for s in cfg.STAGES}
n_gpus = np.array(sorted(times))
total = np.array([sum(times[n].values()) for n in n_gpus])

fig = plt.figure(figsize=cfg.figure_size(height_mm=90))
# Panel (a) is square, (b) takes the rest of the width
grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.25))
ax_speedup = fig.add_subplot(grid[0])
ax_speedup.set_box_aspect(1)
# Panel (b) only holds the donuts
ax_breakdown = fig.add_subplot(grid[1])
ax_breakdown.axis("off")

ax_speedup.plot(n_gpus, n_gpus / n_gpus[0], ls="--", color=cfg.COLORS["guide"], label="Ideal")
ax_speedup.plot(n_gpus, total[0] / total, marker="o", color=cfg.COLORS["jvmc_dense"], label="Measured")
for axis in (ax_speedup.xaxis, ax_speedup.yaxis):
    axis.set_inverted(False)
ax_speedup.set_xscale("log", base=2)
ax_speedup.set_yscale("log", base=2)
# Explicit labels at the numbers of GPUs, the base-10 formatter of plain_log_labels would hide those on a base-2 axis
ax_speedup.set_xticks(n_gpus, labels=n_gpus)
ax_speedup.set_yticks(n_gpus, labels=n_gpus)
ax_speedup.minorticks_off()
ax_speedup.set_xlabel("Number of GPUs")
ax_speedup.set_ylabel("Speedup")
ax_speedup.set_title("(a)", loc="left")
ax_speedup.legend()

# Donuts one above the other, as [x0, y0, width, height] in fractions of panel (b)
for i, n in enumerate(BREAKDOWN_GPUS):
    plot_donut(ax_breakdown.inset_axes([0.2, 0.5 - 0.52 * i, 0.6, 0.54]), times[n], n)

fig.tight_layout()

# The square panel (a) is lower than its slot, so (b) gets its label on the line of (a), above its leftmost text
fig.canvas.draw()
title = ax_speedup.title
labels_left = min(t.get_window_extent().x0 for inset in ax_breakdown.child_axes for t in inset.texts)
x = title.get_transform().inverted().transform((labels_left, 0))[0]
fig.text(x, title.get_position()[1], "(b)", transform=title.get_transform(), va=title.get_va())

cfg.save(fig, "gpu_scaling")
