# tfim_dynamics

Quench of the 2D transverse-field Ising model, from the x-polarized state to h = h_c, with a ResNet and TDVP.
Each folder is one run: its `run.py`, `input.json` and `nets/`, copied unchanged from the encap experiment
`tfim_dynamics/<version>`. The scripts differ from run to run, so each one is kept as it was run.

| Folder | encap version | Data file | Lattice, n_C, samples | Integrator | Step-size norm |
|---|---|---|---|---|---|
| `6x6_nc10/` | `6_by_6` | `jvmc_6x6_nc10.csv` | 6x6, 10, 8e4 | AdaptiveHeun, tol = 1e-4 | `tdvp_norm` |
| `8x8_nc16/` | `Ns_3e5` | `jvmc_8x8_nc16.csv` | 8x8, 16, 3e5 | RK45, atol = 1e-3 | `s_norm_fn` |
| `8x8_nc20/` | `larger_net` | `jvmc_8x8_nc20.csv` | 8x8, 20, 1e5 | RK45, atol = 1e-6 | `tdvp_norm` |
| `10x10_nc16/` | `10_by_10` | `jvmc_10x10_nc16.csv` | 10x10, 16, 3e5 | RK45, atol = 1e-3 | `s_norm_fn` |

To run one, start it from its folder, `cd 8x8_nc16 && python run.py`. It reads `./input.json` and writes
`data_<name>.csv`, `data_<name>.h5` and the prepared initial state `params_xpol_<name>.npy` into that folder. The CSV
is the file copied to `Data/tfim_dynamics/`.

- `6x6_nc10` is an older version of the script: it runs on one GPU, has no checkpointing, and imports
  `jVMC_exp.solver.pinv_snr`. That module was merged into `jVMC_exp.solver.pinv` on 2026-10-05, so with a newer
  jVMC_exp the import has to become `from jVMC_exp.solver.pinv import PinvSNR`. Its CSV header has one column name
  more than there are columns.
- The other three restart from a checkpoint in their HDF5 file when started again, and run on several GPUs with
  `JVMC_USE_DISTRIBUTED=true`.
- In `8x8_nc20` the RK45 tolerance is hard-coded to 1e-6, while `input.json` says `tol = 1e-7`, which is only written
  to the metadata.
