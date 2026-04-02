"""
Process HREMD output: separate state trajectories, run getcontacts,
accumulate contact-frequency data, and write analysis outputs.

Temps and replica count are read from ``chacra_run.json`` when available
so that repeated restarts do not require re-supplying ``--min_temp``,
``--max_temp``, and ``--n_systems``.
"""

import argparse
import gc
import os
import re
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from chacra.ContactFrequencies import make_contact_dataframe, ContactFrequencies
from chacra.trajectories.process_hremd import (
    load_femto_data,
    get_num_states,
    ReplicaHandler,
    get_exchange_probabilities,
    get_exchange_probabilities,
    get_state_energies,
)
import GPUtil
from chacra.plot import plot_chacras, plot_difference_of_roots, plot_explained_variance
from chacra.visualize.pymol import to_pymol
from chacra.utils import RunConfig

import MDAnalysis as mda


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _update_latest_symlink(analysis_dir: str, run: int) -> None:
    """
    Create / update ``analysis_output/latest`` → ``analysis_output/run_N``.
    Uses a relative symlink so the project directory is portable.
    """
    latest = Path(analysis_dir) / "latest"
    target = Path(f"run_{run}")  # relative
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target)


def _sorted_contact_files(freq_dir: str) -> list[str]:
    """Return conditionally sorted contact files (.parquet or .tsv) sorted by state index."""
    all_files = os.listdir(freq_dir)
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    target_files = parquet_files if parquet_files else [f for f in all_files if f.endswith(".tsv")]
    
    return [
        f"{freq_dir}/{f}"
        for f in sorted(
            target_files,
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
        )
    ]


def _accumulate_contacts(
    run: int,
    current_run_df: pd.DataFrame,
    selection_file: str,
) -> pd.DataFrame:
    """
    Merge *current_run_df* with the accumulated contact data from all prior runs.
    Uses MDAnalysis trajectory length to determine weights.
    """
    if run == 1:
        return current_run_df

    prior_parquet = Path(f"./analysis_output/run_{run - 1}/total_contacts.parquet")
    if not prior_parquet.exists():
        print(
            f"  [process-output] No prior parquet found at {prior_parquet}; "
            "falling back to unweighted direct merge."
        )
        # Without tracking headers across all history, a flat fallback is safest.
        # This only triggers if upgrading midway through an old project.
        return current_run_df

    prior_df = pd.read_parquet(prior_parquet)
    
    # Recover frame counts via MDAnalysis (only needs 1 IO call per run)
    current_xtc = f"./state_trajectories/run_{run}/state_0.xtc"
    current_frames = len(mda.Universe(selection_file, current_xtc).trajectory) if os.path.exists(current_xtc) else 1

    prior_total_frames = 0
    for i in range(1, run):
        prior_xtc = f"./state_trajectories/run_{i}/state_0.xtc"
        if os.path.exists(prior_xtc):
            prior_total_frames += len(mda.Universe(selection_file, prior_xtc).trajectory)

    total_frames = prior_total_frames + current_frames

    # Align columns — union with zero-fill
    all_cols = prior_df.columns.union(current_run_df.columns)
    prior_df = prior_df.reindex(columns=all_cols, fill_value=0.0)
    current_run_df = current_run_df.reindex(columns=all_cols, fill_value=0.0)

    # Re-weight
    if prior_total_frames > 0 and total_frames > 0:
        result = (
            prior_df * (prior_total_frames / total_frames)
            + current_run_df * (current_frames / total_frames)
        )
    else:
        result = current_run_df
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process the HREMD output.  Replica trajectories are separated into "
            "individual thermodynamic-state trajectories (state_trajectories/run_N/), "
            "contacts are calculated (contact_output/run_N/), and cumulative analysis "
            "outputs are written to analysis_output/run_N/."
        )
    )
    parser.add_argument(
        "--run", type=int, required=True, help="The run ID to process."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to chacra_run.json.  If present, temps/n_systems are read "
             "from this file and CLI flags below become optional.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of jobs to use for parallel calculations.",
    )
    parser.add_argument(
        "--structure_file",
        type=str,
        default=None,
        help="Path to the topology / structure file.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save trajectory data at this cycle interval.",
    )
    parser.add_argument(
        "--output_selection",
        type=str,
        default="protein",
        help="MDAnalysis atom selection for writing state trajectories.",
    )
    # Legacy / override args — not required when chacra_run.json is present
    parser.add_argument(
        "--min_temp",
        type=float,
        default=None,
        help="Minimum effective temperature (K).  Read from config if omitted.",
    )
    parser.add_argument(
        "--max_temp",
        type=float,
        default=None,
        help="Maximum effective temperature (K).  Read from config if omitted.",
    )
    parser.add_argument(
        "--n_systems",
        type=int,
        default=None,
        help="Number of replicas.  Read from config if omitted.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------- #
    # Resolve configuration                                                   #
    # ---------------------------------------------------------------------- #
    config_path = args.config or (
        "chacra_run.json" if os.path.exists("chacra_run.json") else None
    )
    run_config = RunConfig(config_path)
    run_config.apply_to_namespace(args)

    run = args.run
    structure_file = args.structure_file

    if structure_file is None:
        parser.error(
            "--structure_file is required (or set 'structure_file' in chacra_run.json)."
        )

    # Build the temperature array from the stored list or re-derive it
    if run_config.get("temps") is not None:
        temps = np.array(run_config.get("temps"))
    elif args.min_temp is not None and args.max_temp is not None and args.n_systems is not None:
        temps = np.geomspace(args.min_temp, args.max_temp, args.n_systems)
    else:
        parser.error(
            "Temperature information is missing.  Either provide --config pointing "
            "to chacra_run.json, or supply --min_temp, --max_temp, and --n_systems."
        )

    # ---------------------------------------------------------------------- #
    # Directory setup                                                         #
    # ---------------------------------------------------------------------- #
    os.makedirs(f"./state_trajectories/run_{run}", exist_ok=True)
    os.makedirs(f"./analysis_output/run_{run}", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/contacts", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/freqs", exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Save protein-only reference structure                                   #
    # ---------------------------------------------------------------------- #
    structure_name = re.split(r"\/|\.", structure_file)[-2]
    selection_file = f"./structures/{structure_name}_protein.pdb"
    if not os.path.exists(selection_file):
        protein = mda.Universe(structure_file).select_atoms("protein")
        protein.write(selection_file)

    # ---------------------------------------------------------------------- #
    # Load HREMD state data                                                   #
    # ---------------------------------------------------------------------- #
    hremd_data = f"./replica_trajectories/run_{run}/samples.arrow"
    df = load_femto_data(hremd_data)
    n_states = get_num_states(df)

    # Write the protein-only selection file used as the contact topology
    u = mda.Universe(structure_file)
    u.select_atoms(args.output_selection).write(selection_file)

    # ---------------------------------------------------------------------- #
    # Energy plots                                                            #
    # ---------------------------------------------------------------------- #
    from chacra.plot import plot_energies
    plot_energies(
        get_state_energies(df),
        filename=f"./analysis_output/run_{run}/state_energies.png",
        n_bins=50,
    )

    # ---------------------------------------------------------------------- #
    # State trajectory assembly                                               #
    # ---------------------------------------------------------------------- #
    replica_handler = ReplicaHandler(
        structure=structure_file,
        traj_dir=f"./replica_trajectories/run_{run}/trajectories",
        hremd_data=hremd_data,
        save_interval=args.save_interval,
    )
    replica_handler.write_state_trajectories(
        output_dir=f"./state_trajectories/run_{run}",
        selection=args.output_selection,
        ref=selection_file,
    )

    # ---------------------------------------------------------------------- #
    # Exchange probabilities                                                  #
    # ---------------------------------------------------------------------- #
    exchange_probs = get_exchange_probabilities(df)
    np.save(f"./analysis_output/run_{run}/exchange_probabilities", exchange_probs)
    with open(f"./analysis_output/run_{run}/exchange_probabilities.txt", "w") as f:
        for i, prob in enumerate(exchange_probs):
            f.write(f"{i}\n\t{prob:.4f}\n")

    del replica_handler, df
    gc.collect()

    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Contact calculations (ultracontacts -> fallback getcontacts)           #
    # ---------------------------------------------------------------------- #
    
    gpus = GPUtil.getGPUs()
    n_gpus = len(gpus)
    
    # We fetch system xml config if it exists
    openmm_sys = run_config.get("system_file") or args.system_file

    if n_gpus > 0:
        print(f"[process-output] Detected {n_gpus} GPUs. Running ultracontacts in parallel chunks...")
        # Break states into chunks of size n_gpus
        states = list(range(n_states))
        chunks = [states[i:i + n_gpus] for i in range(0, len(states), n_gpus)]
        
        for chunk in chunks:
            processes = []
            for gpu_offset, state_idx in enumerate(chunk):
                contacts_out = f"contact_output/run_{run}/contacts/cont_state_{state_idx}.parquet"
                freqs_out = f"contact_output/run_{run}/freqs/freqs_state_{state_idx}_condensed.parquet"
                
                cmd = [
                    "ultracontacts", "contacts",
                    "--topology", selection_file,
                    "--trajectory", f"./state_trajectories/run_{run}/state_{state_idx}.xtc",
                    "--output", contacts_out,
                    "--condensed", freqs_out,
                    "--stride", "1",
                ]
                if openmm_sys and os.path.exists(openmm_sys):
                    cmd.extend(["--openmm-system", str(openmm_sys)])
                    
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_offset)
                
                proc = subprocess.Popen(cmd, env=env)
                processes.append(proc)
            
            # Wait for all states in this GPU batch to finish
            for proc in processes:
                proc.wait()
                if proc.returncode != 0:
                    print(f"[process-output] WARNING: An ultracontacts subprocess failed with return code {proc.returncode}.")
                    
    else:
        print("[process-output] No GPUs detected. Falling back to getcontacts (CPU).")
        for i in range(n_states):
            contacts_out = f"contact_output/run_{run}/contacts/cont_state_{i}.tsv"
            freqs_out = f"contact_output/run_{run}/freqs/freqs_state_{i}.tsv"

            subprocess.run(
                [
                    "get-dynamic-contacts",
                    "--topology", selection_file,
                    "--trajectory", f"./state_trajectories/run_{run}/state_{i}.xtc",
                    "--output", str(contacts_out),
                    "--cores", str(args.n_jobs),
                    "--itypes", "all",
                    "--distout",
                    "--sele", "protein",
                    "--sele2", "protein",
                ],
                check=True,
            )

            subprocess.run(
                [
                    "get-contact-frequencies",
                    "--input_files", str(contacts_out),
                    "--output_file", str(freqs_out),
                ],
                check=True,
            )

    # ---------------------------------------------------------------------- #
    # Contact frequency accumulation                                          #
    # ---------------------------------------------------------------------- #
    current_freq_dir = f"./contact_output/run_{run}/freqs"
    current_files = _sorted_contact_files(current_freq_dir)
    current_run_df = make_contact_dataframe(current_files)

    # Save a per-run summary parquet (raw, unweighted) for archival
    current_run_df.to_parquet(
        f"./contact_output/run_{run}/freqs_summary.parquet", index=True
    )

    # Compute (or update) the cumulative weighted contact frequencies
    cdf = _accumulate_contacts(run, current_run_df, selection_file)
    del current_run_df
    gc.collect()

    # Persist the cumulative result — parquet is portable across pandas versions
    cdf.to_parquet(f"./analysis_output/run_{run}/total_contacts.parquet", index=True)

    # ---------------------------------------------------------------------- #
    # ChACRA analysis                                                         #
    # ---------------------------------------------------------------------- #
    cf = ContactFrequencies(cdf, temps=np.round(temps), n_jobs=args.n_jobs)

    top_ten = {
        pc: cf.cpca.sorted_norm_loadings(pc)[f"PC{pc}"][:10].index.tolist()
        for pc in cf.cpca.top_chacras
    }
    pd.DataFrame(top_ten).to_csv(
        f"./analysis_output/run_{run}/top_chacra_contacts.csv", index=False
    )

    fig = plot_chacras(
        cf.cpca,
        n_pcs=cf.cpca.top_chacras[-1],
        contacts=cf.freqs,
        temps=temps,
        temp_scale="K",
        filename=f"./analysis_output/run_{run}/chacra_modes.png",
    )
    fig.clf()

    fig = plot_difference_of_roots(
        cf.cpca,
        n_pcs=cf.cpca.top_chacras[-1],
        filename=f"./analysis_output/run_{run}/difference_of_roots.png",
    )
    fig.clf()

    plot_explained_variance(
        cf.cpca,
        filename=f"./analysis_output/run_{run}/explained_variance.png",
    )

    to_visualize = []
    for pc in cf.cpca.top_chacras:
        to_visualize.extend(cf.cpca.get_chacra_center(pc, cutoff=0.7).index)

    to_pymol(
        to_visualize,
        cf.freqs,
        cf.cpca,
        output_file=f"./analysis_output/run_{run}/top_chacras.pml",
        pc_range=(cf.cpca.top_chacras[0], cf.cpca.top_chacras[-1]),
        variable_sphere_scale=True,
    )

    # ---------------------------------------------------------------------- #
    # Update analysis_output/latest symlink                                  #
    # ---------------------------------------------------------------------- #
    _update_latest_symlink("./analysis_output", run)
    print(f"[process-output] Done.  See analysis_output/run_{run}/ (or analysis_output/latest/).")


if __name__ == "__main__":
    main()
