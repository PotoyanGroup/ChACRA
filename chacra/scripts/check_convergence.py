"""
Standalone CLI for convergence diagnostics.

Usage::

    chacra check-convergence --run 3
    chacra check-convergence --run 3 --bootstrap
    chacra check-convergence --history
"""

import argparse
import os
import sys

import numpy as np

from chacra.convergence import (
    bootstrap_loadings,
    convergence_report,
    plot_convergence_history,
    plot_exchange_diagnostics,
    print_convergence_report,
    save_convergence_report,
)


def _discover_n_states(contact_base: str, run: int, file_pattern: str | None = None) -> int | None:
    """Figure out n_states by counting contact files."""
    from pathlib import Path

    if file_pattern is not None:
        # Count distinct state indices by trying 0, 1, 2... until none match
        base = Path(contact_base)
        n = 0
        while True:
            glob_pat = file_pattern.format(state=n)
            if any(base.glob(glob_pat)):
                n += 1
            else:
                break
        return n if n > 0 else None

    # Default layout: count cont_state_* files in the run's contacts dir
    contacts_dir = os.path.join(contact_base, f"run_{run}", "contacts")
    if not os.path.isdir(contacts_dir):
        return None
    count = 0
    for f in os.listdir(contacts_dir):
        if f.startswith("cont_state_") and (f.endswith(".parquet") or f.endswith(".tsv")):
            count += 1
    return count if count > 0 else None


def _discover_latest_run(contact_base: str) -> int | None:
    """Find the highest run number in contact_base."""
    if not os.path.isdir(contact_base):
        return None
    runs = []
    for d in os.listdir(contact_base):
        if d.startswith("run_") and os.path.isdir(os.path.join(contact_base, d)):
            try:
                runs.append(int(d.replace("run_", "")))
            except ValueError:
                pass
    return max(runs) if runs else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check convergence of ChACRA analysis.  Computes RMSIP from "
            "per-frame contact records, contact correlation, and exchange "
            "diagnostics."
        )
    )
    parser.add_argument(
        "--run", type=int, default=None,
        help="Run number to analyze.  Defaults to the latest run.",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top PCs to compare in RMSIP (default: 3).",
    )
    parser.add_argument(
        "--bootstrap", action="store_true", default=False,
        help="Run bootstrap loading stability analysis (adds ~1-2 min).",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=100,
        help="Number of bootstrap resamples (default: 100).",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help=(
            "Number of parallel workers for loading contact files. "
            "Default 1 (sequential) to avoid doubling peak RAM.  "
            "Increase on machines with ample memory."
        ),
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        metavar="N",
        help=(
            "Subsample each state to at most N evenly-spaced frames before "
            "computing frequencies.  Useful on memory-constrained machines "
            "(e.g. --max_frames 20000 cuts RAM by ~5× for a 100k-frame run)."
        ),
    )
    parser.add_argument(
        "--history", action="store_true", default=False,
        help="Plot convergence history across all runs and exit.",
    )
    parser.add_argument(
        "--contact_base", type=str, default="./contact_output",
        help="Path to contact output root.",
    )
    parser.add_argument(
        "--analysis_dir", type=str, default="./analysis_output",
        help="Path to analysis output directory.",
    )
    parser.add_argument(
        "--file_pattern", type=str, default=None,
        metavar="PATTERN",
        help=(
            "Glob pattern (relative to --contact_base) for per-state contact files.\n"
            "Use {state} as a placeholder for the state index.  Examples:\n"
            "  run_*/rep_{state}_contacts.tsv       (custom naming, multi-run)\n"
            "  contacts/rep_{state}_contacts.tsv    (flat single-run directory)\n"
            "Default: standard ChACRA layout (run_*/contacts/cont_state_{state}.*)"
        ),
    )
    parser.add_argument(
        "--n_states", type=int, default=None,
        help=(
            "Number of thermodynamic states.  Auto-discovered when using the "
            "default layout.  Required (or strongly recommended) when "
            "--file_pattern is specified."
        ),
    )

    args = parser.parse_args()

    # --history mode
    if args.history:
        fig = plot_convergence_history(
            analysis_dir=args.analysis_dir,
            filename=os.path.join(args.analysis_dir, "convergence_history.png"),
        )
        if fig is not None:
            print(f"[check-convergence] History plot saved to "
                  f"{args.analysis_dir}/convergence_history.png")
        return

    # Determine run
    if args.run is not None:
        run = args.run
    else:
        run = _discover_latest_run(args.contact_base)
        if run is None:
            run = 1  # sensible default when using custom file_pattern
            if args.file_pattern is None:
                sys.exit(f"[check-convergence] No runs found in {args.contact_base}/")

    # Determine n_states
    if args.n_states is not None:
        n_states = args.n_states
    else:
        n_states = _discover_n_states(args.contact_base, run, args.file_pattern)
    if n_states is None or n_states == 0:
        msg = (
            f"[check-convergence] No contact files found for run {run}.\n"
        )
        if args.file_pattern:
            msg += f"  Pattern searched: {args.contact_base}/{args.file_pattern.format(state='N')}\n"
            msg += "  Tip: pass --n_states explicitly to skip auto-discovery."
        else:
            msg += f"  Expected: {args.contact_base}/run_{run}/contacts/cont_state_*.{{tsv,parquet}}"
        sys.exit(msg)

    print(f"[check-convergence] Run {run}, {n_states} states, k={args.k}")
    if args.file_pattern:
        print(f"[check-convergence] File pattern: {args.contact_base}/{args.file_pattern}")

    # Load exchange probabilities if available
    exch_path = os.path.join(args.analysis_dir, f"run_{run}", "exchange_probabilities.npy")
    exchange_probs = np.load(exch_path) if os.path.exists(exch_path) else None

    # Compute report
    report = convergence_report(
        run=run,
        n_states=n_states,
        exchange_probs=exchange_probs,
        k=args.k,
        contact_base=args.contact_base,
        file_pattern=args.file_pattern,
        analysis_dir=args.analysis_dir,
        max_frames_per_state=args.max_frames,
        n_jobs=args.n_jobs,
    )

    # Save and print
    out_dir = os.path.join(args.analysis_dir, f"run_{run}")
    os.makedirs(out_dir, exist_ok=True)
    save_convergence_report(report, out_dir)
    print_convergence_report(report)

    # Exchange diagnostics plot
    if exchange_probs is not None:
        fig = plot_exchange_diagnostics(
            exchange_probs,
            filename=os.path.join(out_dir, "exchange_diagnostics.png"),
        )
        fig.clf()
        print(f"  Exchange diagnostics plot: {out_dir}/exchange_diagnostics.png")

    # Convergence history plot
    fig = plot_convergence_history(
        analysis_dir=args.analysis_dir,
        filename=os.path.join(args.analysis_dir, "convergence_history.png"),
    )
    if fig is not None:
        fig.clf()
        print(f"  Convergence history plot:  {args.analysis_dir}/convergence_history.png")

    # Bootstrap (opt-in)
    if args.bootstrap:
        print(f"\n  Running bootstrap loading stability ({args.n_bootstrap} resamples)...")
        stability = bootstrap_loadings(
            n_states=n_states,
            k=args.k,
            n_top=20,
            n_bootstrap=args.n_bootstrap,
            contact_base=args.contact_base,
            file_pattern=args.file_pattern,
            n_jobs=args.n_jobs,
        )
        stability_path = os.path.join(out_dir, "bootstrap_loading_stability.csv")
        stability.to_csv(stability_path)
        print(f"  Bootstrap results saved to: {stability_path}")

        print(f"\n  Top contacts — bootstrap rank stability (top 20, {args.n_bootstrap} samples):")
        for pc in range(1, args.k + 1):
            col = f"PC{pc}_rank_freq"
            if col in stability.columns:
                top = stability[col].nlargest(10)
                stable = (top >= 0.8).sum()
                print(f"    PC{pc}: {stable}/10 contacts appear in top-20 in ≥80% of bootstraps")

    print("[check-convergence] Done.")


if __name__ == "__main__":
    main()
