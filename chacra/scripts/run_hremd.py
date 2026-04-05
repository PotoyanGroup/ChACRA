import argparse
import os
import shutil
import subprocess
import traceback
from datetime import datetime

import numpy as np

from chacra.trajectories.process_hremd import *
from chacra.utils import RunConfig


def _find_mpirun() -> str | None:
    """Return the first mpirun/mpiexec on PATH.

    The conda environment's mpirun takes precedence (appears first on PATH),
    which ensures ABI compatibility with the installed mpi4py.
    """
    return shutil.which("mpirun") or shutil.which("mpiexec")

_CONFIG_PATH = "chacra_run.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run HREMD simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Config file (loaded before remaining args so CLI args can override) #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to chacra_run.json.  Defaults to 'chacra_run.json' in the "
             "current directory if it exists.  All values can be overridden "
             "by the explicit CLI flags below.",
    )

    # Add arguments for system file and structure
    parser.add_argument(
        "-p", "--system_file",
        type=str,
        default=None,
        help="Path to the system XML file.",
    )
    parser.add_argument(
        "-s", "--structure_file",
        type=str,
        default=None,
        help="Path to the structure PDB file.",
    )
    parser.add_argument(
        "-c", "--n_cycles",
        type=int,
        default=None,
        help="The number of replica exchange attempts for this run.",
    )
    parser.add_argument(
        "-j", "--n_jobs",
        type=int,
        default=None,
        help="The number of MPI processes to start.  For simulation this should "
             "equal the number of available GPUs.  The analysis step "
             "(process-output) uses a separate --n_jobs that can be set "
             "independently for CPU-bound contact calculations.",
    )
    parser.add_argument(
        "-d", "--steps_per_cycle",
        type=int,
        default=None,
        help="The number of timesteps between replica exchange attempts.",
    )
    parser.add_argument(
        "-l", "--min_temp",
        type=float,
        default=None,
        help="The minimum effective temperature (K) of the replica exchange "
             "ensemble.  Same as the solvent temperature for all replicas.",
    )
    parser.add_argument(
        "-x", "--max_temp",
        type=float,
        default=None,
        help="The maximum effective temperature (K) of the replica exchange "
             "ensemble.",
    )
    parser.add_argument(
        "-n", "--n_systems",
        type=int,
        default=None,
        help="The number of replicas.",
    )
    parser.add_argument(
        "-i", "--save_interval",
        type=int,
        default=None,
        help="Save trajectory data at this cycle interval.",
    )
    parser.add_argument(
        "-k", "--checkpoint_interval",
        type=int,
        default=None,
        help="Save checkpoints at this cycle interval.",
    )
    parser.add_argument(
        "-w", "--warmup_steps",
        type=int,
        default=None,
        help="Warmup steps before replica exchange begins (first run only).",
    )
    parser.add_argument(
        "-b", "--lambda_selection",
        type=str,
        default=None,
        help="MDAnalysis selection to which lambda scaling is applied.",
    )
    parser.add_argument(
        "--output_selection",
        type=str,
        default=None,
        help="MDAnalysis selection of atoms to write for state trajectories.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Timestep in femtoseconds.  HMR recommended for timesteps > 2 fs.",
    )
    parser.add_argument(
        "-o","--oversubscribe",
        type=int,
        default=1,
        help="The number of replicas to run simultaneously on each GPU. "
             "Default is 1 meaning that if you have 2 GPUs and 4 replicas, "
             "2 replicas will be assigned to each GPU with each one running sequentially. "
             "If oversubscribe is set to 2, then all 4 replicas will run "
             "simultaneously on the 2 GPUs."
    )
    parser.add_argument(
        "--mpi-command",
        type=str,
        default=None,
        dest="mpi_command",
        help="Path to the MPI launcher.  Auto-detected from PATH if not "
             "specified.  The conda environment's mpirun is used by default "
             "(ABI-compatible with the installed mpi4py).  Set this if your "
             "cluster requires a specific launcher, e.g. 'srun --mpi=pmix'.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load chacra_run.json, then fill any args still None from the config #
    # ------------------------------------------------------------------ #
    config_path = args.config or (_CONFIG_PATH if os.path.exists(_CONFIG_PATH) else None)
    run_config = RunConfig(config_path)
    run_config.apply_to_namespace(args)

    # Apply hard-coded defaults for anything still None
    _hard_defaults = {
        "n_cycles": 1000,
        "steps_per_cycle": 1000,
        "min_temp": 290.0,
        "max_temp": 450.0,
        "save_interval": 10,
        "checkpoint_interval": 500,
        "warmup_steps": 0,
        "lambda_selection": "protein",
        "output_selection": "protein",
        "timestep": 2,
    }
    for key, val in _hard_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    # Validate required args
    if args.system_file is None:
        parser.error(
            "--system_file is required (or set 'system_file' in chacra_run.json)."
        )
    if args.structure_file is None:
        parser.error(
            "--structure_file is required (or set 'structure_file' in chacra_run.json)."
        )
    if args.n_jobs is None:
        parser.error(
            "--n_jobs / -j is required."
        )

    current_run = (
        len(
            [
                folder
                for folder in os.listdir("./replica_trajectories")
                if folder.startswith("run")
                and os.path.isdir(f"./replica_trajectories/{folder}")
            ]
        )
        + 1
        if os.path.isdir("./replica_trajectories")
        else 1
    )

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MININTERVAL", "600")  # once every 10 minutes if enabled
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if current_run > 1:
        # Load data from the previous run to verify / infer n_systems
        df = load_femto_data(
            f"replica_trajectories/run_{current_run - 1}/samples.arrow"
        )
        n_systems_from_data = get_num_states(df)
        if args.n_systems is None:
            n_systems = n_systems_from_data
        elif args.n_systems != n_systems_from_data:
            print(
                f"Warning: --n_systems ({args.n_systems}) does not match "
                f"previous runs ({n_systems_from_data}). "
                f"Using {n_systems_from_data}."
            )
            n_systems = n_systems_from_data
        else:
            n_systems = args.n_systems
    else:  # current_run == 1
        if args.n_systems is None:
            parser.error(
                "--n_systems / -n is required on the first run "
                "(or set 'n_systems' in chacra_run.json)."
            )
        n_systems = args.n_systems

    
    # Get the number of cycles run thus far
    # Get the number of cycles run thus far
    if current_run == 1:
        total_cycles = args.n_cycles
    else:
        cycles_completed = (
            int((df["step"].values / args.steps_per_cycle)[-1]) + 1
        )
        total_cycles = cycles_completed + args.n_cycles

    # ------------------------------------------------------------------ #
    # Build run-femto args (run-femto always runs inside MPI now)         #
    # ------------------------------------------------------------------ #
    femto_args = [
        "chacra", "run-femto",
        "--system_file",   args.system_file,
        "--structure_file", args.structure_file,
        "--n_cycles",      str(total_cycles),
        "--steps_per_cycle", str(args.steps_per_cycle),
        "--min_temp",      str(args.min_temp),
        "--max_temp",      str(args.max_temp),
        "--n_systems",     str(n_systems),
        "--save_interval", str(args.save_interval),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--warmup_steps",  str(args.warmup_steps),
        "--lambda_selection", args.lambda_selection,
        "--timestep",      str(args.timestep),
    ]

    # Total MPI ranks = GPUs-in-use × oversubscription factor
    n_total_ranks = args.n_jobs * args.oversubscribe

    # Use the mpirun on PATH (conda env's mpirun takes precedence and is
    # ABI-compatible with the installed mpi4py).  Override with --mpi-command
    # if your cluster requires a specific launcher.
    if args.mpi_command:
        mpirun = args.mpi_command
    else:
        mpirun = _find_mpirun()
    if mpirun is None:
        raise RuntimeError(
            "Cannot find mpirun or mpiexec on PATH. "
            "Install OpenMPI (e.g. sudo apt install openmpi-bin) or add it to PATH."
        )
    print(f"Using mpirun: {mpirun}  ({n_total_ranks} ranks on {args.n_jobs} GPU(s) "
          f"× oversubscribe={args.oversubscribe})")

    mpi_command = [mpirun, "-np", str(n_total_ranks), "--oversubscribe"] + femto_args
    
    
    # Compute and cache the full temperature list so process-output and
    # the JSON config don't need to re-derive it.
    temps = np.geomspace(args.min_temp, args.max_temp, n_systems).tolist()

    # ------------------------------------------------------------------ #
    # Write / update chacra_run.json (first run creates it; later runs    #
    # update current_run and any overridden fields).                      #
    # ------------------------------------------------------------------ #
    run_config.update(
        system_file=args.system_file,
        structure_file=args.structure_file,
        n_systems=n_systems,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        temps=temps,
        n_cycles=args.n_cycles,
        steps_per_cycle=args.steps_per_cycle,
        save_interval=args.save_interval,
        checkpoint_interval=args.checkpoint_interval,
        warmup_steps=args.warmup_steps,
        lambda_selection=args.lambda_selection,
        output_selection=args.output_selection,
        timestep=args.timestep,
        n_jobs=args.n_jobs,
        current_run=current_run,
    )
    run_config.write(_CONFIG_PATH)

    times = {}
    times["start"] = datetime.now().strftime("%H:%M")

    # Prepare MPS-aware environment for oversubscribed runs
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

    try:
        log_dir = Path(f"./analysis_output/run_{current_run}")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "hremd_stdout.log", "wb") as out, \
             open(log_dir / "hremd_stderr.log", "wb") as err:
            result = subprocess.run(
                mpi_command,
                stdout=out,
                stderr=err,
                check=True,
                env=run_env,
            )

        print("Replica exchange completed:", result.returncode)

        # Move femto output files to the run directory
        os.makedirs(f"./replica_trajectories/run_{current_run}")
        shutil.move(
            "./hremd-outputs/trajectories/",
            f"./replica_trajectories/run_{current_run}/",
        )
        shutil.move(
            "./hremd-outputs/samples.arrow",
            f"./replica_trajectories/run_{current_run}/samples.arrow",
        )
        shutil.copy(
            "./hremd-outputs/checkpoint.pkl",
            f"./replica_trajectories/run_{current_run}/checkpoint.pkl",
        )

        times["end"] = datetime.now().strftime("%H:%M")

        # Process the replicas to state trajectories and run analyses.
        # Pass --config so process-output reads temps/n_systems from JSON.
        analysis_command = [
            "chacra", "process-output",
            "--run",
            str(current_run),
            "--n_jobs",
            str(args.n_jobs),
            "--structure_file",
            args.structure_file,
            "--save_interval",
            str(args.save_interval),
            "--output_selection",
            args.output_selection,
            "--config",
            _CONFIG_PATH,
        ]
        subprocess.run(analysis_command, check=False)

        # Update current_run in JSON after successful completion
        run_config.config["current_run"] = current_run
        run_config.write(_CONFIG_PATH)

        # Write human-readable stats
        with open(f"analysis_output/run_{current_run}/stats.txt", "w") as f:
            f.write(f"{times['start']} to {times['end']}\n")
            f.write(f"n_systems : {n_systems}\n")
            f.write(f"temps     : {temps}\n")
            f.write(f"n_steps   : {args.steps_per_cycle * args.n_cycles}\n")
            f.write(f"save_interval : {args.save_interval}\n")
            f.write(f"checkpoint_interval : {args.checkpoint_interval}\n")

    except subprocess.CalledProcessError as e:
        print("Error: The subprocess call failed.")
        print("Return Code:", e.returncode)
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)

    except Exception:
        print("An unexpected error occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
