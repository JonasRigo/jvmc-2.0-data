"""
Time of three TDVP steps of jVMC_exp for a range of batch sizes, or of numbers of Markov chains.

Transverse-field Ising chain (PBC, g = -0.7), three Euler steps of TDVP (diagonal shift 10). Every run is
one point of a scan: it appends a group bc=<batch size> (batch-size scan) or mc=<number of chains>
(Markov-chain scan) to the HDF5 file of the scan, with the timings of the three stages of a step
(sampling, objective function and gradient, solve; first and last step), the memory statistics of every
device and the metadata. The figures use the timings of the last step.

Needs the jVMC_exp of the branch `benchmarks` (OutputManager.flush_memory_info). Every run is started
with srun, one task per GPU, and JVMC_USE_DISTRIBUTED=true. The results in Data/ were made on NVIDIA
A100-SXM4-80GB GPUs (4 per node) with these runs:

Batch-size scan, RWKV, L = 100 (10097 parameters), 32768 samples, 1 GPU
-> Data/batch_size_scan/RWKV_L=100_Ns=32768_gpus=1.hdf5

    # in a job with 1 task and 1 GPU
    for bs in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768; do
        srun python batch_size_and_mc_chains_scan.py -mode batch_size -net RWKV -L 100 \\
            -num_samples 32768 -batch_size $bs
    done

Markov-chain scan, CpxCNN, L = 50 (3268 parameters), 4096 samples, batch size 4096, on 1, 2 and 4 GPUs
-> Data/mc_chains_scan/CpxCNN_L=50_Ns=4096_bs=4096_gpus=<1, 2, 4>.hdf5

    # in a job with 1, 2 or 4 GPUs
    for mc in 10 13 18 25 35 48 66 90 124 170 234 320 439 603 826 1133 1553 2128 2918 4000; do
        srun python batch_size_and_mc_chains_scan.py -mode mc_chains -net CpxCNN -num_samples 4096 \\
            -batch_size 4096 -mc_chains $mc
    done
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
parser.add_argument("-batch_size", default=2048, type=int)
parser.add_argument("-mc_chains", default=100, type=int, help="No effect for autoregressive nets")
parser.add_argument("-net", default="CpxCNN", choices=["CpxCNN", "RWKV", "RNN"])
parser.add_argument("-mode", default="batch_size", choices=["batch_size", "mc_chains"])
parser.add_argument("-L", default=50, type=int, help="Number of spins")
parser.add_argument("-out_dir", default=None, type=Path,
                    help="Default: paper/Data/batch_size_scan or paper/Data/mc_chains_scan")
args = parser.parse_args()


# parameters
L = args.L
g = -0.7
num_samples = args.num_samples
mode = args.mode # "batch_size" or "mc_chains"
net_name = args.net # "CpxCNN", "RWKV" or "RNN"
batch_size = args.batch_size
mc_chains = args.mc_chains # has no effect for autoregressive nets

if net_name in ("RWKV", "RNN") and mode == "mc_chains":
    raise ValueError(f"{net_name} is sampled autoregressively, the mc_chains scan is useless.")


print(f"Running benchmark with batch_size={batch_size}, mc_chains={mc_chains}")

# Initialize net
if net_name == "CpxCNN":
    net_params = {"F": [20,], "channels": [10, 8, 6, 4], "bias": True, "firstLayerBias": True}
    net = nets.CpxCNN(F=net_params["F"], channels=net_params["channels"], bias=net_params["bias"], firstLayerBias=net_params["firstLayerBias"]) # same as in figure 3d of the old paper
elif net_name == "RWKV":
    net_params = {"L": L, "LHilDim": 2, "patch_size": 1, "embedding_size": 16, "num_heads": 3, "num_layers": 3, "hidden_size": 8}
    net = nets.RWKVCPM(L=net_params["L"], LHilDim=net_params["LHilDim"], patch_size=net_params["patch_size"], embedding_size=net_params["embedding_size"], num_heads=net_params["num_heads"], num_layers=net_params["num_layers"], hidden_size=net_params["hidden_size"])
elif net_name == "RNN":
    net_params = {"L": L, "hiddenSize": 40, "depth": 2, "cell": "RNN"}
    net = nets.RNN1DGeneral(L=net_params["L"], hiddenSize=net_params["hiddenSize"], depth=net_params["depth"], cell=net_params["cell"])

psi = jVMC_exp.vqs.NQS(net, L, batchSize=batch_size, seed=1234)
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

# Set up OutputManager, one file per scan and one group per point of the scan
out_dir = args.out_dir or DATA_DIR / f"{mode}_scan"
out_dir.mkdir(parents=True, exist_ok=True)
n_gpus = jax.device_count()
if mode == "batch_size":
    chains = "" if psi.is_generator else f"_chains={mc_chains}"
    output_manager = OutputManager(path=out_dir / f"{net_name}_L={L}_Ns={num_samples}{chains}_gpus={n_gpus}.hdf5", group=f"bc={batch_size}", append=True)
else:
    output_manager = OutputManager(path=out_dir / f"{net_name}_L={L}_Ns={num_samples}_bs={batch_size}_gpus={n_gpus}.hdf5", group=f"mc={mc_chains}", append=True)

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

# Set up TDVP
tdvp = jVMC_exp.optimizer.TDVP(
    sampler,
    psi,
    make_real=True,
    diagonalShift=10,
    output_manager=output_manager
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
    "mode": mode,
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
