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
    get_state_energies,
    freq_frames,
)
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


def _sorted_tsv_files(freq_dir: str) -> list[str]:
    """Return getcontacts TSV files sorted by state index."""
    return [
        f"{freq_dir}/{f}"
        for f in sorted(
            os.listdir(freq_dir),
            key=lambda x: int(re.split(r"_|\.", x)[-2]),
        )
        if f.endswith(".tsv")
    ]


def _accumulate_contacts(
    run: int,
    current_run_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge *current_run_df* with the accumulated contact data from all
    prior runs by reading the previous run's ``total_contacts.parquet``.

    Rather than re-reading all historical TSV files (O(N_runs × N_states)
    files), we store a weighted running total in ``total_contacts.parquet``
    and update it incrementally (O(1) parquet reads regardless of run count).

    The weights are proportional to the number of trajectory frames each
    run contributed, as recorded in the first-row header of any freq TSV.

    Parameters
    ----------
    run : int
        The current run number.
    current_run_df : pd.DataFrame
        Raw (unweighted) contact frequency DataFrame for run *N*.

    Returns
    -------
    pd.DataFrame
        Frame-count-weighted cumulative contact frequency DataFrame.
    """
    if run == 1:
        return current_run_df

    prior_parquet = Path(f"./analysis_output/run_{run - 1}/total_contacts.parquet")
    if not prior_parquet.exists():
        # Fallback: re-derive from all TSVs (handles migration from old output)
        print(
            f"  [process-output] No prior parquet found at {prior_parquet}; "
            "falling back to reading all historical TSV files."
        )
        all_dfs: dict[int, pd.DataFrame] = {}
        all_frame_counts: dict[int, int] = {}
        for i in range(1, run + 1):
            freq_dir = f"./contact_output/run_{i}/freqs"
            tsv_files = _sorted_tsv_files(freq_dir)
            all_frame_counts[i] = freq_frames(tsv_files[0])
            all_dfs[i] = make_contact_dataframe(tsv_files)
        total_frames = sum(all_frame_counts.values())
        weighted = [
            all_dfs[i] * (all_frame_counts[i] / total_frames)
            for i in range(1, run + 1)
        ]
        combined = pd.concat(weighted, axis=0).fillna(0)
        return combined.groupby(combined.index).sum().reset_index(drop=True)

    # Fast incremental path: load prior weighted total + current run
    prior_df = pd.read_parquet(prior_parquet)
    # Recover frame counts for prior total and current run
    current_freq_dir = f"./contact_output/run_{run}/freqs"
    current_tsv_files = _sorted_tsv_files(current_freq_dir)
    current_frames = freq_frames(current_tsv_files[0])

    # Sum of frames in all prior runs.  We store this in the parquet metadata
    # as a fallback we count TSV headers only for runs that exist.
    prior_total_frames = 0
    for i in range(1, run):
        prior_freq_dir = f"./contact_output/run_{i}/freqs"
        prior_tsv = _sorted_tsv_files(prior_freq_dir)
        prior_total_frames += freq_frames(prior_tsv[0])

    total_frames = prior_total_frames + current_frames

    # Align columns — union with zero-fill
    all_cols = prior_df.columns.union(current_run_df.columns)
    prior_df = prior_df.reindex(columns=all_cols, fill_value=0.0)
    current_run_df = current_run_df.reindex(columns=all_cols, fill_value=0.0)

    # Re-weight: prior_df was weighted at prior_total_frames / old_total;
    # we need weights at prior_total_frames / total_frames.
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
    # Contact calculations (getcontacts)                                     #
    # ---------------------------------------------------------------------- #
    for i in range(n_states):
        contacts_out = f"contact_output/run_{run}/contacts/cont_state_{i}.tsv"
        freqs_out = f"contact_output/run_{run}/freqs/freqs_state_{i}.tsv"

        subprocess.run(
            [
                sys.executable,
                Path(__file__).parent.parent.parent
                / "external"
                / "getcontacts"
                / "get_dynamic_contacts.py",
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
                sys.executable,
                Path(__file__).parent.parent.parent
                / "external"
                / "getcontacts"
                / "get_contact_frequencies.py",
                "--input_files", str(contacts_out),
                "--output_file", str(freqs_out),
            ],
            check=True,
        )

    # ---------------------------------------------------------------------- #
    # Contact frequency accumulation                                          #
    # ---------------------------------------------------------------------- #
    current_tsv_files = _sorted_tsv_files(f"./contact_output/run_{run}/freqs")
    current_run_df = make_contact_dataframe(current_tsv_files)

    # Save a per-run summary parquet (raw, unweighted) for archival
    current_run_df.to_parquet(
        f"./contact_output/run_{run}/freqs_summary.parquet", index=True
    )

    # Compute (or update) the cumulative weighted contact frequencies
    cdf = _accumulate_contacts(run, current_run_df)
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
