#!/usr/bin/env python3
"""Quench dynamics of the 2D transverse-field Ising model with NQS (jVMC_exp / jvmc-2.0).

Reproduces the setup of

    M. Schmitt, M. Heyl, "Quantum many-body dynamics in two dimensions with
    artificial neural networks", PRL 125, 100503 (2020), arXiv:1912.08828v3

on an L x L square lattice:

    H = -J sum_<i,j> sigma^z_i sigma^z_j - h sum_i sigma^x_i

The system is prepared in the x-polarized product state |+ + ... +> (the ground
state of the h -> infinity limit) and evolved with H at a transverse field
h = h_ratio * h_c, where h_c = 3.04438 J is the critical field of the 2D
square-lattice TFIM [Bloete & Deng, PRE 66, 066110 (2002)].

RESTARTS
--------
The run checkpoints itself into ``data_<name>.h5`` and picks up where it left
off when it is started again, so a job that hits the cluster walltime can simply
be resubmitted:

    * every ``simulation.checkpoint_every`` steps the flat parameter vector and
      the position in the trajectory (time, step, number of recorded rows) are
      written to the ``checkpoint`` group of the HDF5 file, in place;
    * on startup that record is read back, the state is restored, and the
      observable/metadata datasets are truncated to the number of rows the
      checkpoint covers -- rows written after the last checkpoint belong to a
      state that is gone, so they would otherwise be duplicated;
    * new rows are appended to the same HDF5 file, and the CSV is rewritten from
      the full (restored + new) history, so both stay consistent;
    * the x-polarized state preparation is skipped on a restart.

To make the last step before a kill count, the run also stops cleanly at a step
boundary when it receives SIGTERM (which Slurm sends before SIGKILL) or when
``simulation.max_walltime`` is exhausted. Without an explicit setting the budget
is taken from the remaining Slurm time minus ``simulation.walltime_margin``.

Set ``simulation.resume`` to false to force a fresh start; an existing HDF5 file
that cannot be resumed is never overwritten, it is moved aside with a timestamp
suffix.

Two ansaetze are available:

    * ``CNN``    -- complex-parameter CNN, holomorphic (``nets/cnn.py``)
    * ``ResNet`` -- real-parameter CNN with residual connections and complex
                    output, non-holomorphic (``nets/resnet.py``)

Both are translation invariant by construction (periodic padding plus a sum over
lattice sites in the output), so only the global spin flip is projected
explicitly (``simulation.symmetry``). The initial state is prepared by a
variational ground-state search for H_x = -sum_i sigma^x_i, stopped as soon as
the energy variance density Var(H_x) / N drops below
``simulation.init.var_tol`` (see ``prepare_x_polarized``).

One process handles one entry of ``inp["simulation"]["runs"]``, selected via
ENCAP_PROCID, so the whole sweep (architectures x fields) runs as
``-sl_i <len(runs)>``.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import json
import signal
import subprocess
import time
from datetime import datetime

import h5py
import jax.numpy as jnp
import jVMC_exp
import numpy as np
import pandas as pd
from jax.experimental import multihost_utils
from jVMC_exp.objective_function import Observable
from jVMC_exp.util.util import s_norm_fn
from jVMC_exp.operator.discrete import SigmaX, SigmaZ
from jVMC_exp.solver.pinv import PinvSNR
# from jVMC_exp.solver.pinv import Pinv
from jVMC_exp.util.output_manager import OutputManager
from nets.cnn import CpxCNN2D
from nets.resnet import ResNet

# With JVMC_USE_DISTRIBUTED=true every rank (one per GPU) runs this whole script.
# Only rank 0 writes files and prints. Branches that depend on files, signals or
# the wall clock are decided on rank 0 and broadcast (``all_agree``): a rank that
# takes a different branch never joins the next collective, and the job hangs.
# jax.process_index() is only safe here, after importing jVMC_exp has run
# jax.distributed.initialize().
MAIN = jax.process_index() == 0

if not MAIN:
    def print(*args, **kwargs):
        pass


def all_agree(flag):
    """Rank 0's value of ``flag``, on every rank."""
    return bool(multihost_utils.broadcast_one_to_all(np.int32(flag)))


def load_on_main(path, like):
    """``np.load(path)`` on rank 0, broadcast to all ranks.

    The file is shared with concurrently running jobs, so separate reads on each
    rank could see different versions of it.
    """
    data = np.load(path) if MAIN else np.zeros(like.shape, like.dtype)

    return multihost_utils.broadcast_one_to_all(np.asarray(data, dtype=like.dtype))


i = int(os.environ.get("ENCAP_PROCID", 0))

with open("input.json", "r") as f:
    inp = json.load(f)

sim = inp["simulation"]
run_cfg = sim["runs"][i]

# ------------------------------------------------------------ physical parameters

L = inp["system"]["L"]
J = inp["system"]["J"]
h_crit = inp["system"]["h_crit"]
periodic = inp["system"]["periodic"]
N = L * L

h_ratio = run_cfg["h_ratio"]
h = h_ratio * h_crit

# ---------------------------------------------------------- simulation parameters


def cfg(key, default=None):
    """Run-specific value with fallback to the global simulation block."""
    return run_cfg.get(key, sim.get(key, default))


arch = run_cfg["net"]
net_cfg = dict(inp["net"][arch])
net_cfg.update({k: v for k, v in run_cfg.items() if k in net_cfg})

filterSize = net_cfg["filter_size"]
numChannels = net_cfg["num_channels"]
numLayers = net_cfg["num_layers"]
batchSize = inp["net"]["batchSize"]

tmax = cfg("tmax")
dt = cfg("dt")
integratorTol = cfg("tol")
maxStep = cfg("maxStep")
maxSteps = cfg("max_steps")
invCutoff = cfg("invCutoff")
snrTol = cfg("snr_tol")
diagonalShift = cfg("diagonalShift", 0.0)
seed = cfg("seed")
numSamples = cfg("num_samples")
numChains = cfg("numChains")
thermalizationSweeps = cfg("thermalizationSweeps")
saveMetaData = cfg("save_meta_data", True)
saveParamsEvery = cfg("save_params_every", 0)
symmetry = cfg("symmetry", "spin_flip")

# Restart behaviour.
checkpointEvery = cfg("checkpoint_every", 10)
resumeEnabled = cfg("resume", True)
walltimeMargin = cfg("walltime_margin", 900)

init_cfg = sim["init"]

# The network (and hence the prepared initial state) does not depend on h, so the
# initial parameters are shared by all quenches of the same architecture.
net_name = f"{arch}_L={L}_nc={numChannels}_nl={numLayers}_F={filterSize}_sym={symmetry}"
param_name = f"{net_name}_h={h_ratio}hc_numSamp={numSamples}"
if snrTol != 2:
    param_name += f"_snrTol={snrTol}"

h5_path = f"./data_{param_name}.h5"
csv_path = f"./data_{param_name}.csv"

print(f"ENCAP_PROCID  = {i}", flush=True)
print(f"run config    = {run_cfg}", flush=True)
print(f"name          = {param_name}", flush=True)
print(f"h             = {h_ratio} * h_c = {h}", flush=True)
print(f"devices       = {jax.devices()}", flush=True)


# --------------------------------------------------------------- walltime budget


def parse_duration(value):
    """Seconds from a number or a Slurm duration ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None

    text = str(value).strip()
    if not text or not text[0].isdigit():  # UNLIMITED, INVALID, N/A, ...
        return None

    days, _, clock = text.rpartition("-")
    try:
        parts = [float(p) for p in clock.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:  # 'MM:SS' and 'MM' are left-padded with zeros
        parts.insert(0, 0.0)
    seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    if days:
        seconds += float(days) * 86400

    return seconds


def slurm_time_left():
    """Remaining walltime of the enclosing Slurm job, or None outside Slurm."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%L"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as err:
        print(f"  could not query the Slurm time limit ({err})", flush=True)

        return None

    return parse_duration(out.stdout.strip()) if out.returncode == 0 else None


walltimeBudget = parse_duration(cfg("max_walltime"))
if walltimeBudget is None:
    time_left = slurm_time_left()
    if time_left is not None:
        walltimeBudget = max(time_left - walltimeMargin, 0.0)
        print(
            f"walltime      = {time_left:.0f}s left, stopping after "
            f"{walltimeBudget:.0f}s (margin {walltimeMargin:.0f}s)",
            flush=True,
        )
else:
    print(f"walltime      = stopping after {walltimeBudget:.0f}s", flush=True)

# Slurm sends SIGTERM before SIGKILL; stop at the next step boundary so that the
# checkpoint and the CSV are complete. A second signal terminates immediately.
stop_requested = {"reason": None}


def request_stop(signum, frame):
    stop_requested["reason"] = f"received {signal.Signals(signum).name}"
    signal.signal(signum, signal.SIG_DFL)


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, request_stop)


# ------------------------------------------------------------------ wave function

if arch == "CNN":
    net = CpxCNN2D(
        Lx=L, Ly=L,
        F=(filterSize,) * 2,
        channels=(numChannels,) * numLayers,
        bias=True,
        firstLayerBias=False,
        periodicBoundary=periodic,
    )
elif arch == "ResNet":
    if numChannels % 2 != 0:
        raise ValueError(
            f"The ResNet splits the channels of the last block into real and "
            f"imaginary part, so num_channels must be even. Got {numChannels}."
        )
    net = ResNet(
        Lx=L, Ly=L,
        F=(filterSize,) * 2,
        channels=(numChannels,) * numLayers,
        strides=(1, 1),
        bias=True,
        periodicBoundary=periodic,
    )
else:
    raise ValueError(f"Network architecture '{arch}' is not defined.")

# Both architectures sum over lattice sites, so translation invariance is built
# in and only the global spin flip is projected explicitly. Both the initial
# state and H are invariant under it, so the dynamics stays in that sector.
if symmetry == "spin_flip":
    orbit = jVMC_exp.symmetry.lattice_symetries.spin_flip_symmetry(L, L, "spin")
elif symmetry == "none":
    orbit = None
else:
    raise ValueError(f"Symmetry '{symmetry}' is not defined.")

psi = jVMC_exp.vqs.NQS(net, (L, L), batchSize=batchSize, seed=seed, orbit=orbit)

print(f"parameters    = {psi.numParameters} ({psi.parameters_flat.size} real)", flush=True)
print(f"holomorphic   = {psi.holomorphic}", flush=True)


# ------------------------------------------------------------------- Hamiltonian


def site(x, y):
    """Flat index of lattice site (x, y) in the raveled (L, L) sample."""
    return (y % L) * L + (x % L)


def bonds(dx, dy):
    """Site pairs displaced by (dx, dy); wrapping bonds only for periodic BC."""
    out = []
    for y in range(L):
        for x in range(L):
            if not periodic and (x + dx >= L or y + dy >= L):
                continue
            out.append((site(x, y), site(x + dx, y + dy)))

    return out


nn_bonds = bonds(1, 0) + bonds(0, 1)

# H = -J sum_<ij> sz_i sz_j - h sum_i sx_i
hamiltonian = sum(-J * (SigmaZ(a) * SigmaZ(b)) for a, b in nn_bonds)
hamiltonian = hamiltonian + sum(-h * SigmaX(idx) for idx in range(N))

# Parent Hamiltonian of the initial state: H_x |+ + ... +> = -N |+ + ... +>, so
# the x-polarized product state is its (unique, gapped) ground state and its
# energy variance vanishes there.
H_x = sum(-SigmaX(idx) for idx in range(N))

# Intensive observables: per site, resp. per bond.
d2_bonds = bonds(2, 0) + bonds(0, 2)
observables = {
    "X": sum(SigmaX(idx) for idx in range(N)) / N,
    "Z": sum(SigmaZ(idx) for idx in range(N)) / N,
    "ZZ_nn": sum(SigmaZ(a) * SigmaZ(b) for a, b in nn_bonds) / len(nn_bonds),
    "XX_nn": sum(SigmaX(a) * SigmaX(b) for a, b in nn_bonds) / len(nn_bonds),
    "ZZ_d2": sum(SigmaZ(a) * SigmaZ(b) for a, b in d2_bonds) / len(d2_bonds),
}


# ----------------------------------------------------------------------- sampler

sampler = jVMC_exp.sampler.MCSampler(
    psi,
    jVMC_exp.propose.SpinFlipZ2(),
    key=seed,
    numChains=numChains,
    numSamples=numSamples,
    thermalizationSweeps=thermalizationSweeps,
    sweepSteps=N,
)


# --------------------------------------------------------------------- restarts


def read_restart_record(path):
    """Restart record of a previous run of this configuration, or None.

    Primary source is the ``checkpoint`` group written by ``write_checkpoint``.
    Files from runs that only stored the archival snapshots of
    ``save_params_every`` are still usable: the latest ``parameters/<step>``
    group carries the time of the state it holds, and the number of rows to keep
    follows from the recorded times.
    """
    if not os.path.isfile(path):
        return None

    try:
        with h5py.File(path, "r") as handle:
            group = handle.get("checkpoint")
            if group is not None and "parameters_flat" in group:
                return dict(
                    parameters=np.asarray(group["parameters_flat"][()]),
                    time=float(group.attrs["time"]),
                    step=int(group.attrs["step"]),
                    num_rows=int(group.attrs["num_rows"]),
                    dt=float(group.attrs.get("dt", dt)),
                    elapsed=float(group.attrs.get("elapsed", 0.0)),
                    segment=int(group.attrs.get("segment", 1)),
                    complete=bool(group.attrs.get("complete", False)),
                    source="checkpoint",
                )

            snapshots = handle.get("parameters")
            if not snapshots:
                return None
            name = max(snapshots.keys())
            snapshot = snapshots[name]
            if "time" not in snapshot.attrs:
                return None
            t_snapshot = float(snapshot.attrs["time"])
            times = np.asarray(handle["observables"]["times"][()], dtype=float)

            def read_tree(group):
                return {
                    key: read_tree(item) if isinstance(item, h5py.Group) else item[()]
                    for key, item in group.items()
                }

            return dict(
                parameters=read_tree(snapshot),
                time=t_snapshot,
                step=int(snapshot.attrs.get("step", name)),
                # Rows at or after the snapshot time are re-created by the restart.
                num_rows=int(np.searchsorted(times, t_snapshot, side="left")),
                dt=dt,
                elapsed=0.0,
                segment=1,
                complete=False,
                source=f"parameters/{name}",
            )
    except (OSError, KeyError, ValueError) as err:
        print(f"  WARNING: cannot read a restart record from {path} ({err})", flush=True)

        return None


def truncate_series(handle, name, n_rows):
    """Cut the time series under ``name`` back to ``n_rows`` entries.

    Datasets stored with a fixed shape (or with an integer time axis, which
    earlier versions of this script produced) are recreated as resizable float
    datasets so that the restart can append to them.
    """
    group = handle.get(name)
    if group is None:
        return

    def walk(parent):
        for key in list(parent.keys()):
            item = parent[key]
            if isinstance(item, h5py.Group):
                walk(item)
                continue

            resizable = item.chunks is not None and item.maxshape[0] is None
            wrong_dtype = key == "times" and not np.issubdtype(item.dtype, np.floating)
            if resizable and not wrong_dtype:
                if item.shape[0] > n_rows:
                    item.resize(n_rows, axis=0)
                continue

            data = np.asarray(item[:n_rows])
            if wrong_dtype:
                data = data.astype(float)
            del parent[key]
            parent.create_dataset(
                key, data=data, maxshape=(None,) + data.shape[1:], chunks=True
            )

    walk(group)


def load_series(handle, name):
    """Read a time-series group back into the in-memory layout of OutputManager."""
    group = handle.get(name)
    if group is None:
        return None

    def walk(parent):
        out = {}
        for key, item in parent.items():
            if isinstance(item, h5py.Group):
                out[key] = walk(item)
            else:
                out[key] = [np.asarray(v) for v in item[()]]

        return out

    return walk(group)


restart = read_restart_record(h5_path) if resumeEnabled else None
# Every rank has to be done reading before rank 0 starts writing to the file.
multihost_utils.sync_global_devices("restart record read")

if restart is not None:
    stored = restart["parameters"]
    if isinstance(stored, np.ndarray) and stored.size != psi.parameters_flat.size:
        print(
            f"  WARNING: {h5_path} holds {stored.size} parameters but the network "
            f"has {psi.parameters_flat.size}; ignoring the restart record.",
            flush=True,
        )
        restart = None

if restart is not None and restart["complete"]:
    print(
        f"{h5_path} is already complete (t = {restart['time']:.5f}, tmax = {tmax}). "
        f"Set simulation.resume to false to start over.",
        flush=True,
    )
    raise SystemExit(0)

if MAIN and restart is None and os.path.isfile(h5_path):
    # Never silently overwrite data that cannot be continued.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for path in (h5_path, csv_path):
        if os.path.isfile(path):
            os.replace(path, f"{path}.bak-{stamp}")
            print(f"  moved {path} aside to {path}.bak-{stamp}", flush=True)

outp = OutputManager(h5_path, append=restart is not None)

# Static run configuration goes into its own group: the "metadata" group is used
# by the optimizer for the per-step TDVP diagnostics.
for key, value in dict(
    L=L, J=J, h=h, h_ratio=h_ratio, h_crit=h_crit, periodic=periodic,
    architecture=arch, filter_size=filterSize, num_channels=numChannels,
    num_layers=numLayers, num_parameters=psi.parameters_flat.size,
    num_samples=numSamples, num_chains=numChains, batch_size=batchSize,
    seed=seed, tmax=tmax, dt_init=dt, integrator_tol=integratorTol,
    max_step=maxStep, pinv_cutoff=invCutoff, snr_tol=snrTol,
    diagonal_shift=diagonalShift, init_mode=init_cfg["mode"], symmetry=symmetry,
).items():
    outp.write_dataset(key, value, group="config")


def dump_csv(path):
    """Rewrite the scalar time series of the collected data as a CSV table.

    Array-valued metadata (the SNR and covariance spectra) is only kept in the
    HDF5 file, and the static run metadata is skipped. The restored history of
    earlier job segments is part of ``outp.data``, so the table always covers the
    whole trajectory.
    """
    if not MAIN:
        return

    obs = outp.data.get("observables", {})
    times = list(obs.get("times", []))
    columns = {"time": times}

    def collect(prefix, value):
        if isinstance(value, dict):
            for key, val in value.items():
                collect(f"{prefix}_{key}" if prefix else str(key), val)
            return
        if not isinstance(value, list) or len(value) != len(times):
            return
        entries = [np.asarray(v) for v in value]
        if any(e.size != 1 for e in entries):
            return
        columns[prefix] = [e.item() for e in entries]

    for group in (obs, outp.data.get("metadata", {})):
        for key, value in group.items():
            if key != "times":
                collect(key, value)

    # Explicit column order: restored history comes back from HDF5 in alphabetical
    # order, so without this the layout would depend on whether the run was
    # restarted.
    quantities = ["energy"] + list(observables.keys())
    moments = ["mean", "variance", "MC_error"]

    def column_rank(name):
        for rank, quantity in enumerate(quantities):
            if name == quantity or name.startswith(f"{quantity}_"):
                moment = name[len(quantity) + 1:]
                sub = moments.index(moment) if moment in moments else len(moments)

                return (0, rank, sub, name)

        return (1, 0, 0, name)  # metadata, alphabetically

    order = ["time"] + sorted((c for c in columns if c != "time"), key=column_rank)

    # Atomic, so that a kill during the write cannot leave a truncated table.
    tmp = f"{path}.tmp"
    pd.DataFrame(columns)[order].to_csv(tmp, sep=" ", index=False)
    os.replace(tmp, path)


def write_checkpoint(t, step, num_rows, elapsed, segment, complete=False):
    """Store the state needed to continue this trajectory in a later job.

    The datasets are written in place (no delete-and-recreate), so checkpointing
    often does not grow the file.
    """
    if not MAIN:
        return

    flat = np.asarray(psi.parameters_flat, dtype=float)
    with h5py.File(h5_path, "a") as handle:
        group = handle.require_group("checkpoint")
        if "parameters_flat" in group:
            group["parameters_flat"][...] = flat
        else:
            group.create_dataset("parameters_flat", data=flat)
        group.attrs.update(
            time=float(t), step=int(step), num_rows=int(num_rows),
            dt=float(stepper.dt), elapsed=float(elapsed), segment=int(segment),
            complete=bool(complete),
        )


# ------------------------------------------------------- x-polarized initial state

name_init_params = f"./params_xpol_{net_name}.npy"


class InitConverged(RuntimeError):
    pass


def report_state(tag):
    """Print <X>, <Z> and the energy variance density of H_x for the current state."""
    sampler.sample()
    obs = jVMC_exp.util.measure(
        {"X": observables["X"], "Z": observables["Z"], "H_x": H_x}, sampler
    )
    var = float(obs["H_x"]["variance"]) / N
    print(
        f"  {tag}: <X> = {obs['X']['mean']:+.6f} +/- {obs['X']['MC_error']:.2e}, "
        f"<Z> = {obs['Z']['mean']:+.6f} +/- {obs['Z']['MC_error']:.2e}, "
        f"Var(H_x)/N = {var:.2e}",
        flush=True,
    )

    return var


def prepare_x_polarized():
    """Set the network parameters to (approximately) represent |+ + ... +>.

    Two modes:

    * ``gs_search``: variational ground-state search (imaginary time / SR) for
      H_x = -sum_i sigma^x_i, following ``old_scripts/crit_tfim_2D``. Works for
      both architectures.
    * ``perturbed``: all parameters set to small random values. For the CNN this
      gives log psi = const + O(delta), i.e. the x-polarized state up to
      O(delta^2), which is what the ``perturbed_start_delta=*`` experiments in
      old_scripts use. It does *not* work for the ResNet, whose residual path
      passes the configuration through to the output even at vanishing weights.

    Note that the CNN represents |+ + ... +> exactly at zero kernel weights, but
    there all gradients vanish as well (the activation has vanishing derivative
    at the origin), so the quantum geometric tensor is singular and TDVP cannot
    move. Both modes therefore stay a finite distance away from that point: the
    ground-state search is stopped as soon as the energy variance density
    ``Var(H_x) / N`` drops below ``var_tol`` instead of being run to convergence.
    """
    mode = init_cfg["mode"]
    delta = init_cfg["delta"]

    if mode == "perturbed":
        if arch != "CNN":
            raise ValueError(
                f"init mode 'perturbed' only prepares the x-polarized state for "
                f"the CNN, not for {arch}. Use 'gs_search'."
            )
        rng = np.random.default_rng(seed)
        psi.parameters = jnp.array(delta * rng.normal(size=psi.parameters_flat.size))

        return

    if mode != "gs_search":
        raise ValueError(f"init mode '{mode}' is not defined.")

    gs_outp = OutputManager(h5_path, group="gs_search", append=True)
    gs_opt = jVMC_exp.optimizer.SR(
        sampler, psi,
        diagonalShift=init_cfg["diagonalShift"],
        diagonalScale=init_cfg["diagonalScale"],
        solver=PinvSNR(pinv_cutoff=invCutoff,snr_tol=0, diagonalization_mode="distributed"),
        output_manager=gs_outp,
    )

    def stop_when_converged():
        # The observables belong to the parameters before the pending update, so
        # raising here leaves the state that was just measured in place. The
        # objective function is H_x, hence "energy" is Var(H_x).
        obs = gs_outp.data["observables"]
        var = float(np.asarray(obs["energy"]["variance"][-1])) / N
        step = len(obs["times"])
        if var < init_cfg["var_tol"]:
            raise InitConverged(f"Var(H_x)/N = {var:.2e} after {step} steps")

    try:
        gs_opt.ground_state_search(
            init_cfg["steps"],
            Observable(H_x),
            jVMC_exp.stepper.Euler(timeStep=init_cfg["dt"]),
            observables={"X": observables["X"]},
            callback=[stop_when_converged],
        )
        print(
            f"  WARNING: the ground-state search ran all {init_cfg['steps']} steps "
            f"without reaching Var(H_x)/N < {init_cfg['var_tol']:.1e}.",
            flush=True,
        )
    except InitConverged as converged:
        print(f"  converged: {converged}", flush=True)


if restart is not None:
    print("=== restarting from checkpoint ===", flush=True)
    print(
        f"  {h5_path}: {restart['source']}, t = {restart['time']:.5f}, "
        f"step = {restart['step']}, rows = {restart['num_rows']}, "
        f"dt = {restart['dt']:.2e} (segment {restart['segment'] + 1})",
        flush=True,
    )
    psi.parameters = (
        jnp.asarray(restart["parameters"])
        if isinstance(restart["parameters"], np.ndarray)
        else restart["parameters"]
    )

    if MAIN:
        with h5py.File(h5_path, "a") as handle:
            truncate_series(handle, "observables", restart["num_rows"])
            truncate_series(handle, "metadata", restart["num_rows"])
            # Snapshots beyond the restart point describe states that are gone.
            snapshots = handle.get("parameters")
            if snapshots is not None:
                for name in [k for k in snapshots if int(k) > restart["step"]]:
                    del snapshots[name]

    # All ranks read the truncated history, and rank 0 must not write to the file
    # again before they are done.
    multihost_utils.sync_global_devices("history truncated")
    with h5py.File(h5_path, "r") as handle:
        for name in ("observables", "metadata"):
            restored = load_series(handle, name)
            if restored is not None:
                outp.data[name] = restored
    multihost_utils.sync_global_devices("history restored")

    print(
        f"  restored {len(outp.data.get('observables', {}).get('times', []))} rows "
        f"of history",
        flush=True,
    )
else:
    print("=== preparing x-polarized initial state ===", flush=True)
    if all_agree(os.path.isfile(name_init_params)):
        print(f"  loading {name_init_params}", flush=True)
        psi.parameters = load_on_main(name_init_params, psi.parameters_flat)
    else:
        tic = time.perf_counter()
        prepare_x_polarized()
        print(f"  preparation took {time.perf_counter() - tic:.1f}s", flush=True)

        # Atomic write: the runs of the different quenches share this file and are
        # started concurrently.
        if MAIN:
            tmp = f"./params_xpol_{net_name}.{os.getpid()}.tmp.npy"
            np.save(tmp, np.asarray(psi.parameters_flat))
            os.replace(tmp, name_init_params)

    initVar = report_state("initial state")
    if initVar > init_cfg["var_tol"]:
        print(
            f"WARNING: the prepared state is not well x-polarized "
            f"(Var(H_x)/N = {initVar:.2e} > var_tol = {init_cfg['var_tol']:.1e}). "
            f"Increase simulation.init.steps or adjust simulation.init.dt.",
            flush=True,
        )


# ------------------------------------------------------------------ TDVP evolution

# snr_tol=2 is the usual choice for unweighted psi^2 sampling. It has to be set
# to 0 for importance-weighted samples, where the SNR filter discards the
# correct update.
solver = PinvSNR(pinv_cutoff=invCutoff,snr_tol=0., diagonalization_mode="distributed")
opt = jVMC_exp.optimizer.TDVP(
    sampler, psi,
    # Ignored for the holomorphic CNN; selects the real part of the TDVP
    # equation for the real-parameter ResNet.
    make_real=True,
    diagonalShift=diagonalShift,
    solver=solver,
    output_manager=outp,
    store_qgt=True,
)
stepper = jVMC_exp.stepper.RK45(timeStep=dt, rtol = 0., atol=integratorTol, max_step=maxStep)
objective = Observable(hamiltonian)

if restart is not None:
    # Continue with the step size the previous job had adapted to.
    stepper.dt = min(restart["dt"], maxStep)

def record(t):
    """Store the observables of the current state at time ``t``.

    Mirrors ``AbstractOptimizer._measure_and_store``: ``opt.o_loc`` holds the
    local energies of the first stage of the step that was just taken, i.e. of
    the state at ``t``, and the observables reuse its samples. Times are passed
    as floats so that the HDF5 time axis is created with a floating dtype.
    """
    energy = dict(
        mean=jnp.real(opt.o_loc.mean).squeeze(),
        variance=jnp.real(opt.o_loc.var).squeeze(),
        MC_error=jnp.real(opt.o_loc.error_of_mean).squeeze(),
    )
    outp.write_observables(
        float(t), energy=energy, **jVMC_exp.util.measure(observables, sampler)
    )
    if saveMetaData:
        outp.write_metadata(float(t), **opt.meta_data)


print("=== TDVP time evolution ===", flush=True)

t = restart["time"] if restart is not None else 0.0
step = restart["step"] if restart is not None else 0
rows = restart["num_rows"] if restart is not None else 0
elapsed0 = restart["elapsed"] if restart is not None else 0.0
segment = restart["segment"] + 1 if restart is not None else 1

segment_steps = 0
segment_tic = time.perf_counter()
stop_reason = None

# A checkpoint before the first step lets a job that dies early restart from the
# prepared state instead of redoing the state preparation.
write_checkpoint(t, step, rows, elapsed0, segment)

while t < tmax:
    reason = None
    if stop_requested["reason"] is not None:
        reason = stop_requested["reason"]
    elif walltimeBudget is not None and time.perf_counter() - segment_tic > walltimeBudget:
        reason = f"walltime budget of {walltimeBudget:.0f}s exhausted"
    elif maxSteps and step >= maxSteps:
        # Counted over the whole trajectory, not per job: this guards against a
        # step size that collapsed under Monte-Carlo noise.
        reason = f"max_steps = {maxSteps} reached"
    # The signal and the clock reach the ranks at slightly different moments.
    if all_agree(reason is not None):
        stop_reason = reason
        break

    tic = time.perf_counter()

    new_parameters, dt_used = opt.step(t, stepper, objective, normFunction=s_norm_fn(opt))
    # Reuse the samples of the first stage for the measurement.
    sampler._samples, sampler._logPsi, sampler._weights = opt._sampler_out
    dt_used = float(dt_used)
    opt.meta_data["dt"] = dt_used

    record(t)
    rows += 1
    if saveParamsEvery and step % saveParamsEvery == 0:
        outp.write_parameters(step, psi.parameters, attrs={"time": float(t)})
        # write_parameters also keeps every snapshot in memory.
        outp.data["parameters"].clear()

    obs = outp.data["observables"]
    now = time.perf_counter()
    print(
        f">  t = {t:.5f}  dt = {dt_used:.2e}  "
        f"E/N = {float(obs['energy']['mean'][-1]) / N:+.6f}  "
        f"<X> = {float(obs['X']['mean'][-1]):+.6f}  "
        f"<ZZ_nn> = {float(obs['ZZ_nn']['mean'][-1]):+.6f}  "
        f"tdvp_err = {opt.meta_data.get('tdvp_error', float('nan')):.2e}  "
        f"res = {opt.meta_data.get('residual', float('nan')):.2e}  "
        f"acc = {float(jnp.squeeze(sampler.acceptance_ratio())):.3f}  "
        f"[{now - tic:.2f}s]",
        flush=True,
    )
    dump_csv(csv_path)

    psi.parameters = new_parameters
    t += dt_used
    step += 1
    segment_steps += 1

    if checkpointEvery and step % checkpointEvery == 0:
        write_checkpoint(t, step, rows, elapsed0 + (now - segment_tic), segment)

completed = t >= tmax
if completed:
    # Record the final state; every other row was written before its update.
    sampler.sample()
    record(t)
    rows += 1
    dump_csv(csv_path)

write_checkpoint(
    t, step, rows, elapsed0 + (time.perf_counter() - segment_tic), segment,
    complete=completed,
)

for name, item in outp.timings.items():
    outp.write_dataset(f"timing_{name}", float(item["total"]), group="config")

if completed:
    print(f"done at t = {t:.5f}, wrote {csv_path} and {h5_path}", flush=True)
else:
    print(
        f"STOPPED at t = {t:.5f} / {tmax} after {segment_steps} steps "
        f"({stop_reason}). Checkpoint written -- resubmit the job to continue.",
        flush=True,
    )
