import argparse
import os
import pathlib
import time
from datetime import datetime

import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.rest
import femto.md.utils.mpi
import femto.md.utils.openmm
import MDAnalysis as mda
import mdtop
import numpy as np
import openmm
from openmm import LangevinMiddleIntegrator, XmlSerializer, unit
from openmm.app import PDBFile, Simulation


def main():
    parser = argparse.ArgumentParser(description="Benchmark HREMD simulation.")
    parser.add_argument("--system_file", type=str, required=True, help="Path to the system XML file.")
    parser.add_argument("--structure_file", type=str, required=True, help="Path to the structure PDB file.")
    parser.add_argument("--n_cycles", type=int, default=1000, help="The number of replica_exchange attempts for this run.")
    parser.add_argument("--steps_per_cycle", type=int, default=1000, help="The number of timesteps between replica exchange attempts.")
    parser.add_argument("--min_temp", type=float, default=290)
    parser.add_argument("--max_temp", type=float, default=450)
    parser.add_argument("--n_systems", type=int, required=True, help="The number of replicas.")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lambda_selection", type=str, default="protein")
    parser.add_argument("-d", "--timestep", type=int, default=2)

    parser.add_argument(
        "-j", "--n_jobs", type=int, required=True, 
        help="The number of MPI processes/GPUs to start."
    )
    parser.add_argument(
        "-o", "--oversubscribe", type=int, default=1,
        help="The number of replicas to run simultaneously on each GPU."
    )
    parser.add_argument(
        "--mpi-command", type=str, default=None,
        help="Path to the MPI launcher. Auto-detected from PATH if not specified."
    )
    parser.add_argument("--is-mpi-worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if not args.is_mpi_worker:
        import subprocess
        import shutil
        import sys

        n_total_ranks = args.n_jobs * args.oversubscribe
        mpirun = args.mpi_command or shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun is None:
            raise RuntimeError("Cannot find mpirun on PATH.")
            
        cmd = [mpirun, "-np", str(n_total_ranks), "--oversubscribe", "python", __file__, "--is-mpi-worker"]
        
        # Forward arguments
        cmd.extend(sys.argv[1:])
        
        run_env = os.environ.copy()
        if args.oversubscribe > 1:
            import femto.md.utils.mpi as _fmpi
            thread_pct = max(1, 200 // args.oversubscribe)
            run_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
            print(f"MPS oversubscribe={args.oversubscribe}: "
                  f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={thread_pct}%")
            if not _fmpi.is_mps_running():
                print("Starting CUDA MPS daemon...")
                _fmpi.start_mps()

        print(f"Launching Benchmark: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, env=run_env, check=True)
        except KeyboardInterrupt:
            pass
        return

    femto.md.utils.mpi.divide_gpus()

    # Load system and structure
    system_file = args.system_file
    with open(system_file, "r") as file:
        xml = file.read()
    system = XmlSerializer.deserialize(xml)
    structure_file = args.structure_file

    u = mda.Universe(structure_file)
    indices = u.select_atoms(args.lambda_selection).atoms.ix
    solute_idxs = set(indices)

    temp_min = args.min_temp
    temp_max = args.max_temp
    n_systems = args.n_systems
    warmup_steps = args.warmup_steps
    steps_per_cycle = args.steps_per_cycle
    cycles = args.n_cycles
    save_interval = args.save_interval
    checkpoint_interval = args.checkpoint_interval
    timestep = args.timestep

    # Setup REST
    rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)
    femto.md.rest.apply_rest(system, solute_idxs, rest_config)
    pdb = PDBFile(structure_file)
    structure = mdtop.Topology.from_file(structure_file)

    # Initial dummy integrator just to extract base state
    integrator = LangevinMiddleIntegrator(temp_min, 1 / unit.picosecond, timestep * unit.femtosecond)
    # Make it deterministic
    integrator.setRandomNumberSeed(12345)
    
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # Set velocities explicitly and deterministically
    simulation.context.setVelocitiesToTemperature(temp_min, 12345)

    base_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getForces=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )

    output_dir = pathlib.Path("benchmark-hremd-outputs")
    
    # Since this is a benchmarking run, remove any existing output from a previous benchmark
    # ONLY do this on rank zero (since we are using MPI) to avoid race conditions.
    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        if mpi_comm.rank == 0 and output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            import tempfile
            # Adding a tiny sleep so deleting finishes before workers make it
            time.sleep(0.5)

    temps = list(np.geomspace(temp_min, temp_max, n_systems))
    rest_temperatures = temps * openmm.unit.kelvin
    rest_betas = [
        1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * rest_temperature)
        for rest_temperature in rest_temperatures
    ]

    states = [
        {femto.md.rest.REST_CTX_PARAM: rest_beta / rest_betas[0]}
        for rest_beta in rest_betas
    ]
    states = [
        femto.md.utils.openmm.evaluate_ctx_parameters(state, system)
        for state in states
    ]

    intergrator_config = femto.md.config.LangevinIntegrator(
        timestep=timestep * openmm.unit.femtosecond,
    )
    final_integrator = femto.md.utils.openmm.create_integrator(
        intergrator_config, rest_temperatures[0]
    )
    # Deterministic operations
    final_integrator.setRandomNumberSeed(12345)

    simulation = femto.md.utils.openmm.create_simulation(
        system,
        structure,
        coords=base_state,
        integrator=final_integrator,
        state=states[0],
        platform=femto.md.constants.OpenMMPlatform.CUDA,
    )

    hremd_config = femto.md.config.HREMD(
        n_warmup_steps=warmup_steps,
        n_steps_per_cycle=steps_per_cycle,
        n_cycles=cycles,
        trajectory_interval=save_interval,
        checkpoint_interval=checkpoint_interval,
    )

    # Monkey-patch _propose_swaps to capture exchange time cleanly
    original_propose_swaps = femto.md.hremd._propose_swaps
    exchange_time = 0.0

    def timed_propose_swaps(*bargs, **kwargs):
        nonlocal exchange_time
        t_start = time.time()
        res = original_propose_swaps(*bargs, **kwargs)
        t_end = time.time()
        exchange_time += (t_end - t_start)
        return res

    femto.md.hremd._propose_swaps = timed_propose_swaps

    # Initialize clock
    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        mpi_comm.barrier()
        start_time = time.time()

        if mpi_comm.rank == 0:
            print("==========================================")
            print("      STARTING HREMD BENCHMARK            ")
            print("==========================================")
            print(f"Start Time         : {datetime.now().strftime('%H:%M:%S')}")
            print(f"Replicas           : {n_systems}")
            print(f"Cycles             : {cycles}")
            print(f"Steps/Cycle        : {steps_per_cycle}")
            print(f"Total Steps/Replica: {cycles * steps_per_cycle + warmup_steps}")
            print(f"Warmup Steps       : {warmup_steps}")
            print("==========================================\n")

        # R U N   H R E M D
        femto.md.hremd.run_hremd(
            simulation,
            states,
            hremd_config,
            output_dir=output_dir,
        )

        mpi_comm.barrier()
        end_time = time.time()

        if mpi_comm.rank == 0:
            total_time_sec = end_time - start_time
            total_time_hr = total_time_sec / 3600.0
            
            # Aggregate ns across all systems
            # ns/system = (total_steps * timestep_fs) / 1e6
            total_steps_per_sys = cycles * steps_per_cycle + warmup_steps
            ns_per_system = total_steps_per_sys * timestep / 1e6
            total_aggregate_ns = ns_per_system * n_systems

            # Throughput 
            ns_per_hour_single_replica = ns_per_system / total_time_hr
            aggregate_ns_per_hour = total_aggregate_ns / total_time_hr
            
            print("\n==========================================")
            print("      HREMD BENCHMARK RESULTS             ")
            print("==========================================")
            print(f"End Time                     : {datetime.now().strftime('%H:%M:%S')}")
            print(f"Total Wall-clock Time (sec)  : {total_time_sec:.3f} s")
            print(f"Total Wall-clock Time (min)  : {total_time_sec/60.0:.3f} m")
            print(f"Exchange Proposal Time       : {exchange_time:.3f} s  ({(exchange_time/total_time_sec)*100:.1f}%)")
            print("")
            print(f"Total ns per replica         : {ns_per_system:.6f} ns")
            print(f"Total aggregate ns           : {total_aggregate_ns:.6f} ns")
            print(f"Throughput (ns/day/replica)  : {(ns_per_hour_single_replica*24):.2f} ns/day")
            print(f"Aggregate Throughput (ns/day): {(aggregate_ns_per_hour*24):.2f} ns/day")
            print("==========================================")


if __name__ == "__main__":
    main()
