"""
chacra response -- compute single-state cumulant response from a finished run.

Reads chacra_run.json + femto samples + per-state ultracontacts parquet to
produce R^{(1)} (and optionally R^{(2)}) at one reference replica, and
optionally compares to the PCA-derived PC1 from the same project.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from chacra.response import (
    estimate_response_from_chacra_run,
    compare_to_pca,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-state ChACRA response: estimate R^(1) (and optionally R^(2)) "
            "from one reference replica using equilibrium cumulants, and compare "
            "to PC1 if a fitted ContactFrequencies is available."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project_dir", type=str, default=".",
        help="ChACRA project root (must contain chacra_run.json).",
    )
    parser.add_argument(
        "--run", type=int, default=None,
        help="HREMD run number. Default: latest run found.",
    )
    parser.add_argument(
        "--state_idx", type=int, default=None,
        help="Reference replica index. Default: middle of the ladder.",
    )
    parser.add_argument(
        "--method", type=str, default="central",
        choices=["central", "forward", "backward"],
        help="Finite-difference rule for dU/dk at the reference state.",
    )
    parser.add_argument(
        "--no_R2", action="store_true",
        help="Skip the R^(2) (third-cumulant) estimate.",
    )
    parser.add_argument(
        "--save_interval", type=int, default=None,
        help="Contact-frame spacing (cycles) for THIS run. Required when "
             "scoring an older run whose interval differs from the one in "
             "chacra_run.json (which always reflects the latest run).",
    )
    parser.add_argument(
        "--compare_pca", action="store_true",
        help=(
            "Also load total_contacts.parquet, fit ContactFrequencies, and "
            "report |cos(PC1, R^(1))|."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write response_run_N_state_K.{csv,json}. "
             "Default: analysis_output/run_N/.",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="How many top-|R| contacts to print and write.",
    )
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    if not (project / "chacra_run.json").exists():
        sys.exit(
            f"chacra response: no chacra_run.json found in {project}. "
            "Run from a ChACRA project root."
        )

    if args.run is None:
        runs = sorted(
            int(p.name.split("_")[1])
            for p in (project / "replica_trajectories").glob("run_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        if not runs:
            sys.exit("chacra response: no replica_trajectories/run_* found.")
        args.run = runs[-1]

    print(f"[chacra response] project: {project}")
    print(f"[chacra response] run: {args.run}, state_idx: {args.state_idx} "
          f"(None → midpoint), method: {args.method}")

    result = estimate_response_from_chacra_run(
        project_dir=project,
        run=args.run,
        state_idx=args.state_idx,
        method=args.method,
        save_interval=args.save_interval,
    )
    if args.no_R2:
        result.R2 = None

    print(f"\n[chacra response] estimated on {result.n_frames} frames "
          f"at state {result.state_idx} (β = {result.beta} in femto reduced units)")
    for k, v in result.diagnostics.items():
        print(f"  {k:25s}: {v}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        project / "analysis_output" / f"run_{args.run}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    csv_path = out_dir / f"response_state_{result.state_idx}.csv"
    df.to_csv(csv_path)
    print(f"\n[chacra response] wrote {csv_path} ({len(df)} contacts)")

    print(f"\nTop {args.topk} contacts by |R^(1)|:")
    top = result.topk(args.topk, by="R1")
    for name, row in top.iterrows():
        line = f"  {name:40s}  R1={row['R1']:+.4e}"
        if "R2" in row:
            line += f"  R2={row['R2']:+.4e}"
        print(line)

    cmp_summary = None
    if args.compare_pca:
        print("\n[chacra response] comparing to PCA from total_contacts.parquet ...")
        from chacra.ContactFrequencies import ContactFrequencies

        tc_path = project / "analysis_output" / f"run_{args.run}" / "total_contacts.parquet"
        if not tc_path.exists():
            sys.exit(f"  missing {tc_path} — run process-output first.")
        cdf = pd.read_parquet(tc_path)
        from chacra.utils import RunConfig
        cfg = RunConfig(str(project / "chacra_run.json"))
        temps = np.array(cfg.get("temps"))
        cf = ContactFrequencies(cdf, temps=np.round(temps), n_jobs=1)

        cmp_summary = compare_to_pca(result.R1, cf, result.contact_names)
        print(f"  |cos(PC1, R^(1))|  : {abs(cmp_summary['cos_pc1_R1']):.4f}  ★")
        if cmp_summary.get("cos_pc2_R1") is not None:
            print(f"  |cos(PC2, R^(1))|  : {abs(cmp_summary['cos_pc2_R1']):.4f}  (should be small)")
        print(f"  aligned contacts   : {cmp_summary['n_aligned']}")
        print(f"  R^(1)-only contacts: {cmp_summary['n_only_in_R1']}")
        print(f"  PCA-only contacts  : {cmp_summary['n_only_in_pca']}")

    summary_path = out_dir / f"response_state_{result.state_idx}_summary.json"
    summary = {
        "run": args.run,
        "state_idx": result.state_idx,
        "method": args.method,
        "n_frames": result.n_frames,
        "beta": result.beta,
        "diagnostics": result.diagnostics,
        "n_contacts": len(result.contact_names),
    }
    if cmp_summary is not None:
        summary["cos_pc1_R1"] = float(cmp_summary["cos_pc1_R1"])
        summary["cos_pc2_R1"] = (
            float(cmp_summary["cos_pc2_R1"])
            if cmp_summary.get("cos_pc2_R1") is not None else None
        )
        summary["n_aligned"] = cmp_summary["n_aligned"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[chacra response] wrote {summary_path}")


if __name__ == "__main__":
    main()
