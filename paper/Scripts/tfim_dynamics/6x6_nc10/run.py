"""Quench dynamics of the 2D transverse-field Ising model with NQS (jVMC_exp / jvmc-2.0).

Reproduces the setup of

    M. Schmitt, M. Heyl, "Quantum many-body dynamics in two dimensions with
    artificial neural networks", PRL 125, 100503 (2020), arXiv:1912.08828v3

on an L x L square lattice with periodic boundary conditions:

    H = -J sum_<i,j> sigma^z_i sigma^z_j - h sum_i sigma^x_i

The system is prepared in the x-polarized product state |+ + ... +> (the ground
state of the h -> infinity limit) and evolved with H at a transverse field
h = h_ratio * h_c, where h_c = 3.04438 J is the critical field of the 2D
square-lattice TFIM [Bloete & Deng, PRE 66, 066110 (2002)]. The three quenches
of interest are h_ratio = 0.1 (ferromagnetic side), 1 (critical) and 2
(paramagnetic side).

Two ansaetze are compared:

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

Output per run:
    * ``data_<name>.h5``  -- observables, per-step TDVP metadata (tdvp_error,
      residual, SNR spectrum, ...) and optional parameter checkpoints, written
      incrementally by the ``OutputManager``.
    * ``data_<name>.csv`` -- the scalar subset of the same data, rewritten after
      every step so that partial results survive a walltime kill.
    * ``params_xpol_<net_name>.npy`` -- prepared x-polarized initial parameters,
      reused by later runs and reruns.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import json
import time

import jax.numpy as jnp
import jVMC_exp
import numpy as np
import pandas as pd
from jVMC_exp.objective_function import Observable
from jVMC_exp.operator.discrete import SigmaX, SigmaZ
from jVMC_exp.solver.pinv_snr import PinvSNR
from jVMC_exp.util.output_manager import OutputManager
from nets.cnn import CpxCNN2D
from nets.resnet import ResNet

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

init_cfg = sim["init"]

# The network (and hence the prepared initial state) does not depend on h, so the
# initial parameters are shared by all quenches of the same architecture.
net_name = f"{arch}_L={L}_nc={numChannels}_nl={numLayers}_F={filterSize}_sym={symmetry}"
param_name = f"{net_name}_h={h_ratio}hc_numSamp={numSamples}"
if snrTol != 2:
    param_name += f"_snrTol={snrTol}"

print(f"ENCAP_PROCID  = {i}", flush=True)
print(f"run config    = {run_cfg}", flush=True)
print(f"name          = {param_name}", flush=True)
print(f"h             = {h_ratio} * h_c = {h}", flush=True)
print(f"devices       = {jax.devices()}", flush=True)


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


# ------------------------------------------------------------------ data output

outp = OutputManager(f"./data_{param_name}.h5", append=False)

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
    HDF5 file, and the static run metadata is skipped.
    """
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

    pd.DataFrame(columns).to_csv(path, sep=" ", index=False)


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

    gs_outp = OutputManager(f"./data_{param_name}.h5", group="gs_search", append=True)
    gs_opt = jVMC_exp.optimizer.SR(
        sampler, psi,
        diagonalShift=init_cfg["diagonalShift"],
        diagonalScale=init_cfg["diagonalScale"],
        solver=PinvSNR(snr_tol=init_cfg["snr_tol"], pinv_cutoff=invCutoff),
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


print("=== preparing x-polarized initial state ===", flush=True)
if os.path.isfile(name_init_params):
    print(f"  loading {name_init_params}", flush=True)
    psi.parameters = jnp.load(name_init_params)
else:
    tic = time.perf_counter()
    prepare_x_polarized()
    print(f"  preparation took {time.perf_counter() - tic:.1f}s", flush=True)

    # Atomic write: the runs of the different quenches share this file and are
    # started concurrently.
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
solver = PinvSNR(snr_tol=snrTol, pinv_cutoff=invCutoff)
opt = jVMC_exp.optimizer.TDVP(
    sampler, psi,
    # Ignored for the holomorphic CNN; selects the real part of the TDVP
    # equation for the real-parameter ResNet.
    make_real=True,
    diagonalShift=diagonalShift,
    solver=solver,
    output_manager=outp,
)
stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=dt, tol=integratorTol, maxStep=maxStep)
objective = Observable(hamiltonian)


def tdvp_norm(v):
    """Fubini-Study norm of a parameter update, ||v||_S^2 = |Re <v, S v>|.

    The default Euclidean parameter norm is dominated by the (nearly) null
    directions of the quantum geometric tensor, which makes the adaptive step
    controller collapse dt under Monte-Carlo noise.
    """
    if psi.holomorphic:
        v = opt._make_cmplx_fn(v)
    Sv = opt._S0(v) if callable(opt._S0) else opt._S0.dot(v)

    return jnp.abs(jnp.real(jnp.vdot(v, Sv)))


csv_path = f"./data_{param_name}.csv"
state = {"step": 0, "tic": time.perf_counter()}


class MaxStepsReached(RuntimeError):
    pass


def after_step():
    """Called by ``time_evolution`` after the observables of a step are stored."""
    step = state["step"]
    obs = outp.data["observables"]
    t = obs["times"][-1]
    now = time.perf_counter()

    if saveParamsEvery and step % saveParamsEvery == 0:
        outp.write_parameters(step, psi.parameters, attrs={"time": float(t)})

    dump_csv(csv_path)

    print(
        f">  t = {t:.5f}  dt = {opt.meta_data.get('dt', float('nan')):.2e}  "
        f"E/N = {float(obs['energy']['mean'][-1]) / N:+.6f}  "
        f"<X> = {float(obs['X']['mean'][-1]):+.6f}  "
        f"<ZZ_nn> = {float(obs['ZZ_nn']['mean'][-1]):+.6f}  "
        f"tdvp_err = {opt.meta_data.get('tdvp_error', float('nan')):.2e}  "
        f"res = {opt.meta_data.get('residual', float('nan')):.2e}  "
        f"acc = {float(jnp.squeeze(sampler.acceptance_ratio())):.3f}  "
        f"[{now - state['tic']:.2f}s]",
        flush=True,
    )

    state["step"] = step + 1
    state["tic"] = now

    if maxSteps and state["step"] >= maxSteps:
        raise MaxStepsReached(
            f"max_steps = {maxSteps} reached at t = {t:.5f} < tmax = {tmax}. The "
            f"adaptive stepper is most likely fighting Monte-Carlo noise."
        )


print("=== TDVP time evolution ===", flush=True)
try:
    opt.time_evolution(
        tmax,
        objective,
        stepper,
        observables,
        callback=[after_step],
        save_meta_data=saveMetaData,
        normFunction=tdvp_norm,
    )
except MaxStepsReached as stop:
    print(f"WARNING: {stop}", flush=True)

dump_csv(csv_path)
outp.write_parameters(state["step"], psi.parameters, attrs={"final": True})
for name, item in outp.timings.items():
    outp.write_dataset(f"timing_{name}", float(item["total"]), group="config")

# jVMC_exp.optimizer.base.time_evolution starts with an integer t = 0, so the
# HDF5 "times" datasets are created with integer dtype and every later time is
# truncated to zero. The in-memory copy holds the correct floats, so the
# datasets are rewritten here.
for group in ("observables", "metadata"):
    times = outp.data.get(group, {}).get("times")
    if times is not None:
        outp.write_dataset("times", np.asarray(times, dtype=float), group=group)

print(f"done, wrote {csv_path} and ./data_{param_name}.h5", flush=True)
