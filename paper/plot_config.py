"""Plotting configuration shared by all paper figures.

Every plotting script starts with

    import plot_config as cfg
    cfg.setup()

Use ``cfg.setup(grid=False)`` for panels with custom grids. Typography,
physical figure sizes, method colors, formatters, paths and PDF exports live
here. LaTeX with Computer Modern and dvipng is required for text rendering.
Include figures at native size to preserve 10 pt labels and 9 pt tick labels.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import ticker

MM = 1 / 25.4
# Measured with the document class/options in main.tex: textwidth = 510 TeX pt.
TEXT_WIDTH_MM = 510 / 72.27 * 25.4
FONTSIZE = 10
TICK_FONTSIZE = 9


def figure_size(*, width_fraction=1.0, height_mm=None):
    """Size for a fraction of the paper text width, without export cropping."""
    width_mm = TEXT_WIDTH_MM * width_fraction
    if height_mm is None:
        height_mm = width_mm * 0.65
    return width_mm * MM, height_mm * MM

PAPER_DIR = Path(__file__).resolve().parent
DATA_DIR = PAPER_DIR / "Data"
FIGURES_DIR = PAPER_DIR / "Figures"

# jVMC_exp in greens, reference methods in purple
COLORS = {
    "vmc": "#30ca92",
    "vmc_light": "#5fc9a3",
    "fci": "#8d82f4",
    "jvmc_dense": "#1d8a63",
    "jvmc_batched": "#30ca92",
    "netket": "#8d82f4",
    "netket_onthefly": "#5b4cc4",
    "tdvp": "#30ca92",
    "tdvp_large": "#1d8a63",
    "exact": "#8d82f4",
    "cnn_chen": "#5b4cc4",
    "cnn_schmitt": "#b58df2",
    # jaxmg comparison
    "jaxmg_off": "#8d82f4",
    "jaxmg_on": "#30ca92",
    # Numbers of GPUs for the mc chains scan
    "gpus_1": "#30ca92",
    "gpus_2": "#8d82f4",
    "gpus_4": "#1d8a63",
    # Stages of a TDVP step
    "objective": "#30ca92",
    "solve": "#8d82f4",
    "sampling": "#1d8a63",
    # Reference lines and their annotations
    "guide": "#808080",
}

STAGES = {
    "compute objective function and gradient": ("Objective function and gradient", COLORS["objective"]),
    "solve": ("Solve", COLORS["solve"]),
    "sampling": ("Sampling", COLORS["sampling"]),
}

def setup(*, grid=True):
    """Apply paper typography; grid=False leaves panels to set their own grids."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "figure.titlesize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.axisbelow": True,
        "axes.formatter.useoffset": False,
        "grid.linewidth": 0.5,
        "hatch.linewidth": 0.45,
        "savefig.bbox": None,
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": grid,
        "grid.linestyle": "--" if grid else "-",
        "grid.alpha": 0.5 if grid else 1.0,
    })


class _PlainLogFormatter(ticker.LogFormatter):
    def __call__(self, x, pos=None):
        return f"{x:g}" if super().__call__(x, pos) else ""

def plain_log_labels(axis):
    # Plain numbers (0.5 rather than 5e-01) on a log axis, also at some minor ticks when it spans up to 3 decades
    axis.set_major_formatter(_PlainLogFormatter(labelOnlyBase=False))
    axis.set_minor_formatter(_PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(3, 0.5)))

def save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
