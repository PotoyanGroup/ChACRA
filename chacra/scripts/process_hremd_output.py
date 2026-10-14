"""
Process HREMD output: contacts on replica trajectories, per-frame occupancy
store, frequencies, and ChACRA analysis.

Pipeline stages
───────────────
1. Strip replicas        → replica_trajectories/run_N/protein/r*_protein.xtc
                           (protein-only copies of femto's full-system DCDs)
2. Exchange statistics   → analysis_output/run_N/exchange_probabilities.npy
3. Contact calculations  → contact_output/run_N/contacts/cont_replica_*.{parquet,tsv}
                           (one ultracontacts/getcontacts run per REPLICA)
4. Occupancy store       → contact_output/run_N/occupancy.npz
                           (frames assigned to thermodynamic states with the
                           exchange record; per-state freqs written alongside)
5. ChACRA analysis       → analysis_output/run_N/ (plots, .pml, total_contacts,
                           convergence report)

Contact occupancy is a property of a configuration, not of the state label,
so contacts are computed once per replica and *relabelled* — no state
trajectories are needed for the analysis.  ``--write_state_trajectories``
additionally demultiplexes chronological per-state XTCs (for visualisation)
from the stripped replicas.

Stages 1–4 are skipped when their outputs exist; once a stage is found
incomplete every downstream stage re-runs (cascade rule).  Stage 5 always
re-runs.  ``--force`` reruns everything.
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

from chacra.ContactFrequencies import ContactFrequencies
from chacra.trajectories.process_hremd import (
    load_femto_data,
    get_exchange_probabilities,
    get_state_energies,
    strip_replica_trajectories,
    demux_state_trajectories,
)
from chacra.plot import plot_chacras, plot_difference_of_roots, plot_explained_variance
from chacra.convergence import (
    convergence_report,
    print_convergence_report,
    save_convergence_report,
    plot_convergence_history,
    plot_exchange_diagnostics,
)
from chacra.visualize.pymol import to_pymol
from chacra.run_config import RunConfig

import MDAnalysis as mda


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _update_latest_symlink(analysis_dir: str, run: int) -> None:
    """``analysis_output/latest`` → ``analysis_output/run_N`` (relative symlink)."""
    latest = Path(analysis_dir) / "latest"
    target = Path(f"run_{run}")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target)


def _replica_contact_file(run: int, replica: int) -> str | None:
    for ext in ("parquet", "tsv"):
        p = f"contact_output/run_{run}/contacts/cont_replica_{replica}.{ext}"
        if os.path.exists(p):
            return p
    return None


def _n_gpus() -> int:
    try:
        import GPUtil

        return len(GPUtil.getGPUs())
    except Exception:  # noqa: BLE001 — GPUtil missing / no driver
        return 0


def _run_weight(run: int) -> float:
    """
    Weight of a run for cumulative averaging: number of exchange cycles read
    from the femto samples table (falls back to the frame count of the
    occupancy store, then 0).
    """
    samples = Path(f"./replica_trajectories/run_{run}/samples.arrow")
    if samples.exists():
        import pyarrow

        with pyarrow.OSFile(str(samples), "rb") as fh, \
             pyarrow.RecordBatchStreamReader(fh) as reader:
            return float(sum(batch.num_rows for batch in reader))
    store = Path(f"./contact_output/run_{run}/occupancy.npz")
    if store.exists():
        with np.load(store, allow_pickle=True) as z:
            return float(z["phi"].shape[1])
    return 0.0


def _accumulate_contacts(run: int, current_run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge this run's frequencies with the cumulative frequencies of all prior
    runs, weighting each run by its number of exchange cycles (simulated
    time), so runs saved at different intervals contribute equally per ns.
    """
    if run == 1:
        return current_run_df
    prior_parquet = Path(f"./analysis_output/run_{run - 1}/total_contacts.parquet")
    if not prior_parquet.exists():
        print(f"  [process-output] No prior parquet at {prior_parquet}; using this run only.")
        return current_run_df
    prior_df = pd.read_parquet(prior_parquet)
    w_cur = _run_weight(run)
    w_prior = sum(_run_weight(i) for i in range(1, run))
    total = w_prior + w_cur
    all_cols = prior_df.columns.union(current_run_df.columns)
    prior_df = prior_df.reindex(columns=all_cols, fill_value=0.0)
    current_run_df = current_run_df.reindex(columns=all_cols, fill_value=0.0)
    if w_prior > 0 and total > 0:
        return prior_df * (w_prior / total) + current_run_df * (w_cur / total)
    return current_run_df


def _run_contacts(
    run: int,
    replicas: list[int],
    topology: str,
    trajectories: list[str],
    itypes: list[str],
    openmm_system: str | None,
    n_jobs: int,
) -> None:
    """Stage 3 worker: ultracontacts (GPU, parquet) or getcontacts (CPU, tsv)."""
    n_gpus = _n_gpus()
    out_dir = f"contact_output/run_{run}/contacts"
    if n_gpus > 0:
        print(f"  Engine: ultracontacts ({n_gpus} GPUs), itypes: {' '.join(itypes)}")
        for start in range(0, len(replicas), n_gpus):
            chunk = replicas[start:start + n_gpus]
            procs = []
            for gpu, r in enumerate(chunk):
                cmd = [
                    "ultracontacts", "contacts",
                    "--topology", topology,
                    "--trajectory", trajectories[r],
                    "--output", f"{out_dir}/cont_replica_{r}.parquet",
                    "--stride", "1",
                    "--itypes", *itypes,
                ]
                if openmm_system and os.path.exists(openmm_system):
                    cmd += ["--openmm-system", str(openmm_system)]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                procs.append((r, subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.PIPE)))
            for r, proc in procs:
                _, err = proc.communicate()
                if proc.returncode != 0:
                    print(f"  [WARN] ultracontacts failed for replica {r} "
                          f"(exit {proc.returncode}). {err.decode().strip()[-500:]}")
    else:
        print("  Engine: getcontacts (CPU) — no GPU detected. This writes large TSVs; "
              "prefer running process-output on a GPU node.")
        gc_itypes = ["all"] if "all" in itypes else itypes
        for r in replicas:
            try:
                subprocess.run(
                    ["get-dynamic-contacts",
                     "--topology", topology, "--trajectory", trajectories[r],
                     "--output", f"{out_dir}/cont_replica_{r}.tsv",
                     "--cores", str(max(1, n_jobs)), "--itypes", *gc_itypes,
                     "--sele", "protein", "--sele2", "protein"],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"  [WARN] getcontacts failed for replica {r} (exit {e.returncode}).")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process HREMD output: strip replica trajectories to protein, compute "
            "contacts per replica, assign frames to thermodynamic states with the "
            "exchange record, and run the ChACRA analysis.  Stages with existing "
            "outputs are skipped; --force reruns everything."
        )
    )
    parser.add_argument("--run", type=int, required=True, help="The run ID to process.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force all stages to re-run, ignoring existing outputs.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to chacra_run.json (default: ./chacra_run.json if present).")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallel workers for stripping/demux/contacts (CPU).")
    parser.add_argument("--structure_file", type=str, default=None,
                        help="Full-system topology/structure PDB used for the simulation.")
    parser.add_argument("--system_file", type=str, default=None,
                        help="OpenMM system XML (H-bond topology for ultracontacts).")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Trajectory save interval (cycles) of THIS run. Default: chacra_run.json.")
    parser.add_argument("--contact_itypes", type=str, default=None,
                        help="Space-separated ultracontacts interaction types ('all', or e.g. "
                             "'hb sb vdw'). Default: chacra_run.json value, else 'all'.")
    parser.add_argument("--output_selection", type=str, default="protein",
                        help="MDAnalysis selection kept in stripped replica / state trajectories.")
    parser.add_argument("--write_state_trajectories", action="store_true", default=False,
                        help="Also demultiplex chronological per-state XTCs (visualisation only; "
                             "not needed for the analysis).")
    parser.add_argument("--skip_convergence", action="store_true", default=False,
                        help="Skip the convergence report.")
    # Legacy overrides — not required when chacra_run.json is present
    parser.add_argument("--min_temp", type=float, default=None)
    parser.add_argument("--max_temp", type=float, default=None)
    parser.add_argument("--n_systems", type=int, default=None)
    args = parser.parse_args()

    # ---------------------------------------------------------------------- #
    # Resolve configuration                                                   #
    # ---------------------------------------------------------------------- #
    config_path = args.config or ("chacra_run.json" if os.path.exists("chacra_run.json") else None)
    run_config = RunConfig(config_path)
    run_config.apply_to_namespace(args)

    run = args.run
    structure_file = args.structure_file
    if structure_file is None:
        parser.error("--structure_file is required (or set 'structure_file' in chacra_run.json).")
    save_interval = args.save_interval or run_config.get("save_interval")
    if save_interval is None:
        parser.error("--save_interval is required (or set 'save_interval' in chacra_run.json).")
    save_interval = int(save_interval)

    if run_config.get("temps") is not None:
        temps = np.array(run_config.get("temps"), dtype=float)
    elif args.min_temp is not None and args.max_temp is not None and args.n_systems is not None:
        temps = np.geomspace(args.min_temp, args.max_temp, args.n_systems)
        run_config.update(temps=temps.tolist())
    else:
        parser.error("Temperature information missing: provide --config or "
                     "--min_temp/--max_temp/--n_systems.")
    n_states = len(temps)

    contact_itypes = args.contact_itypes or run_config.get("contact_itypes") or "all"
    if isinstance(contact_itypes, str):
        contact_itypes = contact_itypes.split()
    run_config.update(contact_itypes=" ".join(contact_itypes), save_interval=save_interval)
    openmm_sys = args.system_file or run_config.get("system_file")
    n_jobs = args.n_jobs if args.n_jobs and args.n_jobs > 0 else None

    # ---------------------------------------------------------------------- #
    # Directories / protein-only reference structure                          #
    # ---------------------------------------------------------------------- #
    rep_dir = f"./replica_trajectories/run_{run}"
    protein_dir = f"{rep_dir}/protein"
    os.makedirs(f"./analysis_output/run_{run}", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/contacts", exist_ok=True)
    os.makedirs(f"./contact_output/run_{run}/freqs", exist_ok=True)

    structure_name = re.split(r"\/|\.", structure_file)[-2]
    selection_file = f"./structures/{structure_name}_protein.pdb"
    if not os.path.exists(selection_file):
        os.makedirs("./structures", exist_ok=True)
        mda.Universe(structure_file).select_atoms(args.output_selection).write(selection_file)

    samples = f"{rep_dir}/samples.arrow"
    if not os.path.exists(samples):
        sys.exit(f"  [ERROR] {samples} not found.")

    # ---------------------------------------------------------------------- #
    # Stage plan (cascade rule)                                               #
    # ---------------------------------------------------------------------- #
    protein_xtcs = [f"{protein_dir}/r{i}_protein.xtc" for i in range(n_states)]
    stage1_done = all(os.path.exists(p) for p in protein_xtcs)
    stage2_done = os.path.exists(f"./analysis_output/run_{run}/exchange_probabilities.npy")
    missing_contacts = [r for r in range(n_states) if _replica_contact_file(run, r) is None]
    stage3_done = not missing_contacts
    store_path = f"./contact_output/run_{run}/occupancy.npz"
    stage4_done = os.path.exists(store_path)

    cascade = args.force
    run1 = cascade or not stage1_done; cascade = cascade or run1
    run2 = cascade or not stage2_done; cascade = cascade or run2
    run3 = cascade or not stage3_done; cascade = cascade or run3
    run4 = cascade or not stage4_done

    # ---------------------------------------------------------------------- #
    # Stage 1: strip replicas to protein                                      #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 1/5: Strip replica trajectories → {protein_dir}")
    if run1:
        protein_xtcs = strip_replica_trajectories(
            structure_file, f"{rep_dir}/trajectories", protein_dir, n_states,
            selection=args.output_selection, n_jobs=n_jobs, overwrite=args.force,
        )
        print(f"  [DONE] {n_states} protein-only replica trajectories.")
    else:
        print(f"  [SKIP] all {n_states} present.")

    # ---------------------------------------------------------------------- #
    # Stage 2: exchange statistics                                            #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 2/5: Exchange statistics")
    if run2:
        df = load_femto_data(samples)
        from chacra.plot import plot_energies

        plot_energies(get_state_energies(df),
                      filename=f"./analysis_output/run_{run}/state_energies.png", n_bins=50)
        exchange_probs = get_exchange_probabilities(df)
        np.save(f"./analysis_output/run_{run}/exchange_probabilities", exchange_probs)
        with open(f"./analysis_output/run_{run}/exchange_probabilities.txt", "w") as f:
            for i, prob in enumerate(exchange_probs):
                f.write(f"{i}\n\t{prob:.4f}\n")
        del df
        gc.collect()
        print("  [DONE] exchange probabilities saved.")
    else:
        print("  [SKIP] exists.")

    # ---------------------------------------------------------------------- #
    # Stage 3: contacts per replica                                           #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 3/5: Contacts per replica")
    if run3:
        todo = list(range(n_states)) if (args.force or stage3_done) else missing_contacts
        if len(todo) < n_states:
            print(f"  [SKIP] {n_states - len(todo)}/{n_states} replicas already have contacts.")
        _run_contacts(run, todo, selection_file, protein_xtcs, contact_itypes,
                      openmm_sys, args.n_jobs if args.n_jobs > 0 else os.cpu_count())
        still = [r for r in range(n_states) if _replica_contact_file(run, r) is None]
        if still:
            sys.exit(f"  [ERROR] contact files missing for replicas {still}; fix and rerun.")
        print("  [DONE] contact calculations complete.")
    else:
        print(f"  [SKIP] all {n_states} replica contact files present.")

    # ---------------------------------------------------------------------- #
    # Stage 4: occupancy store + per-state frequencies                        #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 4/5: Occupancy store")
    from chacra.reweight import OccupancyStore, build_occupancy_store

    if run4:
        n_frames = len(mda.Universe(selection_file, protein_xtcs[0]).trajectory)
        store = build_occupancy_store(".", run, save_interval=save_interval, n_frames=n_frames)
        store.save(store_path)
        print(f"  [DONE] {store.n_states} states x {store.n_frames} frames x "
              f"{store.n_contacts} contacts → {store_path}")
    else:
        store = OccupancyStore.load(store_path)
        print(f"  [SKIP] loaded {store_path}")

    current_run_df = store.freqs()
    current_run_df.index = range(n_states)
    current_run_df.to_parquet(f"./contact_output/run_{run}/freqs_summary.parquet", index=True)
    for s in range(n_states):  # per-state files for tooling that expects them
        current_run_df.iloc[[s]].reset_index(drop=True).to_parquet(
            f"./contact_output/run_{run}/freqs/freqs_state_{s}_condensed.parquet")
    r2s = None
    if args.write_state_trajectories:
        from chacra.reweight import load_femto_reduced_potentials

        _, r2s = load_femto_reduced_potentials(samples, save_interval)
        out = f"./state_trajectories/run_{run}"
        print(f"  Demultiplexing chronological state trajectories → {out}")
        demux_state_trajectories(selection_file, protein_xtcs, r2s[:store.n_frames], out,
                                 ref=selection_file, n_jobs=n_jobs)
    del store
    gc.collect()

    # ---------------------------------------------------------------------- #
    # Stage 5: ChACRA analysis                                                #
    # ---------------------------------------------------------------------- #
    print(f"\n[process-output] Stage 5/5: ChACRA analysis")
    cdf = _accumulate_contacts(run, current_run_df)
    cdf.to_parquet(f"./analysis_output/run_{run}/total_contacts.parquet", index=True)

    cf = ContactFrequencies(cdf, temps=np.round(temps), n_jobs=args.n_jobs)
    top_ten = {pc: cf.cpca.sorted_norm_loadings(pc)[f"PC{pc}"][:10].index.tolist()
               for pc in cf.cpca.top_chacras}
    pd.DataFrame(top_ten).to_csv(f"./analysis_output/run_{run}/top_chacra_contacts.csv", index=False)

    n_pcs = cf.cpca.top_chacras[-1] if cf.cpca.top_chacras else 3
    fig = plot_chacras(cf.cpca, n_pcs=n_pcs, contacts=cf.freqs, temps=temps, temp_scale="K",
                       filename=f"./analysis_output/run_{run}/chacra_modes.png")
    fig.clf()
    fig = plot_difference_of_roots(cf.cpca, n_pcs=n_pcs,
                                   filename=f"./analysis_output/run_{run}/difference_of_roots.png")
    fig.clf()
    plot_explained_variance(cf.cpca, filename=f"./analysis_output/run_{run}/explained_variance.png")

    to_visualize = []
    for pc in cf.cpca.top_chacras:
        to_visualize.extend(cf.cpca.get_chacra_center(pc, cutoff=0.7).index)
    if to_visualize:
        to_pymol(to_visualize, cf.freqs, cf.cpca,
                 output_file=f"./analysis_output/run_{run}/top_chacras.pml",
                 pc_range=(cf.cpca.top_chacras[0], cf.cpca.top_chacras[-1]),
                 variable_sphere_scale=True)

    if not args.skip_convergence:
        exch_path = f"./analysis_output/run_{run}/exchange_probabilities.npy"
        exchange_probs = np.load(exch_path) if os.path.exists(exch_path) else None
        report = convergence_report(
            run=run, n_states=n_states, exchange_probs=exchange_probs,
            k=len(cf.cpca.top_chacras) if cf.cpca.top_chacras else 3,
            max_frames_per_state=5000, n_jobs=args.n_jobs,
        )
        save_convergence_report(report, f"./analysis_output/run_{run}")
        print_convergence_report(report)
        if exchange_probs is not None:
            fig = plot_exchange_diagnostics(
                exchange_probs, filename=f"./analysis_output/run_{run}/exchange_diagnostics.png")
            fig.clf()
        fig = plot_convergence_history(filename="./analysis_output/convergence_history.png")
        if fig is not None:
            fig.clf()

    _update_latest_symlink("./analysis_output", run)
    run_config.write()
    print(f"\n[process-output] Done.  See analysis_output/run_{run}/ (or analysis_output/latest/).")
    print("  Next: `chacra response --run", run, "` for MBAR response functions.")


if __name__ == "__main__":
    main()
