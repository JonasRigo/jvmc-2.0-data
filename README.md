# jvmc-2.0-data

Data, scripts and figures for the jVMC 2.0 paper.

## Layout

```
paper/
├── Scripts/          # produce the data
├── Data/             # results of the scripts
├── Figures/          # PDFs made by the plot_*.py scripts
├── plot_config.py    # style shared by all figures
└── plot_*.py         # one script per figure
```

The libraries and versions are in `requirements.txt` (`pip install -r requirements.txt`).

| Script | Data | Figures |
|---|---|---|
| `compare_netket.py` | `compare_netket/` | `netket_comparison.pdf` |
| `compare_netket_sr.py` | `compare_netket_sr/`, `compare_netket_sr_4gpus/` | `netket_comparison_sr_*.pdf` |
| `compare_netket_minsr.py` | `compare_netket_minsr/`, `compare_netket_minsr_4gpus/` | `netket_comparison_minsr_*.pdf` |
| `compare_netket_sr.py` + `compare_netket_minsr.py` | both of the above | `netket_comparison_sr_minsr_*.pdf` |
| `quantum_dot_2d.py` | `quantum_dot_2d.h5` | `quantum_dot_2d.pdf` |
| `plot_mixed_precision_benchmark.py` | `precision_benchmark.h5` | `mixed_precision_benchmark.pdf` |

- **compare_netket**: time of sampling and of the estimate of ⟨H⟩, jVMC_exp vs NetKet.
- **compare_netket_sr**: time and peak memory of one SR step for a growing number of samples.
- **compare_netket_minsr**: time and peak memory of one MinSR step for a growing number of parameters.
- **quantum_dot_2d**: VMC ground state of 6 electrons in a 2D harmonic trap, compared with FCI.

The NetKet comparisons were run on NVIDIA GH200 GPUs (1 GPU, and 4 GPUs for the `_4gpus` folders)

## Plot typography and physical size

`paper/plot_config.py` owns the shared typography, sizes, method colors, formatters, paths and PDF export helper. Scripts apply it through `cfg.setup()`: Computer Modern text and math via LaTeX, 10 pt labels/legends and 9 pt tick labels. The default setup enables dashed grids; use `cfg.setup(grid=False)` for figures that configure their own grids, including the mixed-precision benchmark. A TeX installation with Computer Modern and `dvipng` is required in addition to the Python requirements.

`plot_config.figure_size()` returns dimensions in inches based on the current one-column RevTeX text width (510 TeX pt, about 179.24 mm). For example, `cfg.figure_size(height_mm=100)` creates a full-width figure. The combined one-GPU SR/MinSR plot uses `width_fraction=0.8` and the quantum-dot plot uses `0.9`, matching their manuscript inclusion widths. Include exported PDFs at these native sizes to preserve the font sizes; avoid cropping the exported canvas with `bbox_inches="tight"`.

From this repository root, regenerate a figure with, for example, `python paper/plot_netket_comparison_sr_minsr_1gpu.py`. Absolute script paths also work from other directories. Outputs go to `paper/Figures/`; the shared module has no dependency on the surrounding manuscript checkout. After changing typography, refresh any copies of these figures in the manuscript's `Figures/` directory.

## Mixed-precision benchmark

Run `python paper/plot_mixed_precision_benchmark.py` to reproduce the accepted four-panel figure, including the mirrored runtime/batch bars, symmetry patterns, precision colors and shared GPU titles. The script reads `paper/Data/precision_benchmark.h5` and writes `paper/Figures/mixed_precision_benchmark.pdf` using `cfg.save(fig, "mixed_precision_benchmark")`. To select another compatible archive, pass `--data /path/to/precision_benchmark.h5`. No Jupyter session or GPU computation is needed.

The bundled HDF5 is an unchanged copy of the original summary archive. It includes RTX 5090 timing measurements, selected batches, A100 job runtimes, independent energy/variance estimates, uncertainty samples and provenance. The manuscript's precision notebook imports the script's `make_figure()` function, so both entry points use one drawing implementation. The former `summary/global_plot_defs.py` import is retained as a compatibility wrapper around `plot_config.setup(grid=False)`.
