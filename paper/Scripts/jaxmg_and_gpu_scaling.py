"""
Time of three TDVP steps of jVMC_exp on several GPUs, with the SR equation solved on one device or
distributed over all devices with jaxmg.

Transverse-field Ising chain (PBC, g = -0.7), three Euler steps of TDVP (diagonal shift 10) with the
PinvSNR solver. Every run writes one HDF5 file with the timings of the three stages of a step (sampling,
objective function and gradient, solve; first and last step), the memory statistics of every device and
the metadata. The figures use the timings of the last step.

The results in Data/ were made on NVIDIA A100-SXM4-80GB GPUs (4 per node) with these runs, all with the
 allocator recommended by jaxmg:

    export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
    export XLA_CLIENT_MEM_FRACTION=0.99

jaxmg scan, RNN (depth 5) of growing hidden size, L = 50, 4096 samples, batch size 1024, 16 GPUs
-> Data/jaxmg_scan/RNN_L=50_Ns=4096_bs=1024_hs=<hidden size>_gpus=16_jaxmg=<off, on>.hdf5
(without jaxmg, hidden size 95 runs out of memory)

    # with 16 GPUs
    for hs in 14 20 28 40 56 80 90; do
        srun python jaxmg_and_gpu_scaling.py -net RNN -num_samples 4096 -batch_size 1024 -RNN_hiddenSize $hs -jaxmg 0
    done
    for hs in 20 28 40 56 80 90 100 110 126; do
        srun python jaxmg_and_gpu_scaling.py -net RNN -num_samples 4096 -batch_size 1024 -RNN_hiddenSize $hs -jaxmg 1
    done

GPU scaling, RNN (depth 5, hidden size 20, 2082 parameters), L = 50, 524288 samples, batch size 1024
per GPU, without jaxmg, on 1, 2, 4, 8 and 16 GPUs
-> Data/gpu_scaling/RNN_L=50_Ns=524288_bs=1024_hs=20_gpus=<1, 2, 4, 8, 16>_jaxmg=off.hdf5

    # with 1, 2, 4, 8 or 16 GPUs
    srun python jaxmg_and_gpu_scaling.py -net RNN -num_samples 524288 -batch_size 1024 -RNN_hiddenSize 20 \\
        -jaxmg 0 -out_dir ../Data/gpu_scaling
"""
import os
from pathlib import Path
import argparse

# Can be set to false to run on a single machine without srun
os.environ.setdefault("JVMC_USE_DISTRIBUTED", "true")

import jax

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
from jVMC_exp.objective_function import Observable
from jVMC_exp.util.output_manager import OutputManager

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-num_samples", default=10000, type=int)
parser.add_argument("-batch_size", default=500, type=int, help="Batch size per device")
parser.add_argument("-mc_chains", default=200, type=int, help="No effect for autoregressive nets")
parser.add_argument("-net", default="RNN", choices=["CpxCNN", "RWKV", "RNN"])
parser.add_argument("-jaxmg", default=0, type=int, help="1: solve distributed over all devices with jaxmg, 0: on one device")
parser.add_argument("-T_A", default=256, type=int, help="Tile size of jaxmg")
parser.add_argument("-RNN_hiddenSize", default=100, type=int)
parser.add_argument("-L", default=50, type=int, help="Number of spins")
parser.add_argument("-out_dir", default=DATA_DIR / "jaxmg_scan", type=Path, help="Default: paper/Data/jaxmg_scan")
args = parser.parse_args()

L = args.L
g = -0.7
num_samples = args.num_samples
batch_size = args.batch_size
mc_chains = args.mc_chains
net_name = args.net # "CpxCNN", "RWKV" or "RNN"

print(f"Running {net_name} benchmark with Ns={num_samples}, batch_size={batch_size}, mc_chains={mc_chains}, jaxmg={args.jaxmg}")


# Initialize net
if net_name == "CpxCNN":
    net_params = {"F": [20,], "channels": [10, 8, 6, 4], "bias": True, "firstLayerBias": True}
    net = nets.CpxCNN(F=net_params["F"], channels=net_params["channels"], bias=net_params["bias"], firstLayerBias=net_params["firstLayerBias"]) # same as in figure 3d of the old paper
elif net_name == "RWKV":
    net_params = {"L": L, "LHilDim": 2, "patch_size": 1, "embedding_size": 16, "num_heads": 3, "num_layers": 3, "hidden_size": 8}
    net = nets.RWKVCPM(L=net_params["L"], LHilDim=net_params["LHilDim"], patch_size=net_params["patch_size"], embedding_size=net_params["embedding_size"], num_heads=net_params["num_heads"], num_layers=net_params["num_layers"], hidden_size=net_params["hidden_size"])
elif net_name == "RNN":
    net_params = {"L": L, "hiddenSize": args.RNN_hiddenSize, "depth": 5, "cell": "RNN"}
    net = nets.RNN1DGeneral(L=net_params["L"], hiddenSize=net_params["hiddenSize"], depth=net_params["depth"], cell=net_params["cell"])

psi = jVMC_exp.vqs.NQS(net, L, batchSize_per_device=batch_size, seed=1234)
print(f"The variational ansatz has {psi.numParameters} parameters.")

# Set up TFI hamiltonian
hamiltonian = 0
for site in range(L):
    hamiltonian += -1.0 * op.SigmaZ(site) * op.SigmaZ((site + 1) % L)
    hamiltonian += g * op.SigmaX(site)
objective_function = Observable(hamiltonian)

# Set up sampler
if psi.is_generator: # Autoregressive nets
    sampler = jVMC_exp.sampler.MCSampler(
        psi,
        updateProposer=None,
        key=jax.random.PRNGKey(4321),
        numSamples=num_samples,
    )
else:
    sampler = jVMC_exp.sampler.MCSampler(
        psi,
        updateProposer=jVMC_exp.propose.SpinFlip(),
        key=jax.random.PRNGKey(4321),
        numChains=mc_chains,
        sweepSteps=L,
        numSamples=num_samples,
        thermalizationSweeps=20,
    )

# Set up OutputManager
args.out_dir.mkdir(parents=True, exist_ok=True)
hidden = f"_hs={args.RNN_hiddenSize}" if net_name == "RNN" else ""
chains = "" if psi.is_generator else f"_chains={mc_chains}"
name = f"{net_name}_L={L}_Ns={num_samples}_bs={batch_size}{chains}{hidden}_gpus={jax.device_count()}_jaxmg={'on' if args.jaxmg else 'off'}"
output_manager = OutputManager(path=args.out_dir / f"{name}.hdf5")

def print_timings(outp: OutputManager):
    total = 0.0
    for key, value in outp.timings.items():
        latest = float(value.get("last_timing", 0.0))
        average = float(value["total"]) / max(int(value["count"]), 1)
        print("\taverage and latest timings of ", key)
        print("\t", average)
        print("\t", latest)
        total += latest
    print("\t=== Total: ", total)

# Set up solver
if args.jaxmg:
    solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="distributed", T_A=args.T_A)
else:
    solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="device")

# Set up TDVP
tdvp = jVMC_exp.optimizer.TDVP(
    sampler,
    psi,
    make_real=True,
    diagonalShift=10,
    output_manager=output_manager,
    solver=solver
)

stepper = jVMC_exp.stepper.Euler(timeStep=1e-2)  # ODE integrator

# Run a few steps of TDVP
for _ in range(3):
    new_parameters, _ = tdvp.step(0, stepper, objective_function)
    psi.parameters = new_parameters

    print("Benchmarking data")
    print_timings(tdvp.output_manager)

tdvp.output_manager.flush_timings()
tdvp.output_manager.flush_memory_info()

metadata = {
    "net_name": net_name,
    "L": L,
    "g": g,
    "num_samples": num_samples,
    "batch_size": batch_size,
    "mc_chains": mc_chains,
    "mode": "jaxmg" if args.jaxmg else "no_jaxmg",
    "net_params": net_params,
    "num_params": psi.numParameters,
}
tdvp.output_manager.write_metadata(**metadata)

# Print memory info
for i, device in enumerate(jax.devices()):
    if jax.process_index() == i:
        try:
            memory_info = device.memory_stats()
            print(f"Device {i}: {memory_info}")
        except:
            print(f"Device {i}: Memory stats not available")
