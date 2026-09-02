"""
chacra response -- contact response functions R^(1), R^(2) from a finished run.

Two estimators:

* ``--estimator mbar`` (default): pools the frames of ALL states with MBAR
  weights (femto stores every configuration's reduced potential under every
  state) and evaluates d<phi>/dlambda, d2<phi>/dlambda2 at any lambda —
  including lambda = 1, the physical (unscaled) Hamiltonian.  Requires the
  per-frame occupancy store (built and cached on first use from the
  ultracontacts parquet files).
* ``--estimator single``: the original single-replica cumulant at one state,
  kept for comparison.

The comparison block scores each estimate against the LOCAL finite-difference
slope of the frequency ladder at the same lambda (the quantity the estimator
actually targets) and reports where it lies in the PC1-PC3 frame of ChACRA's
PCA.  PC1 is the ladder-averaged slope, not the local one, so |cos(PC1, R1)|
is reported but is not a convergence criterion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _latest_run(project: Path) -> int:
    runs = sorted(
        int(p.name.split("_")[1])
        for p in (project / "replica_trajectories").glob("run_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
        and (p / "samples.arrow").exists()
    )
    if not runs:
        sys.exit("chacra response: no replica_trajectories/run_*/samples.arrow found.")
    return runs[-1]


def _pca_loadings(project: Path, run: int, contacts, n_pcs: int = 3):
    """Signed PC loadings from the run's cumulative frequencies, aligned to contacts."""
    tc_path = project / "analysis_output" / f"run_{run}" / "total_contacts.parquet"
    if not tc_path.exists():
        return None
    from sklearn.decomposition import PCA

    tc = pd.read_parquet(tc_path)
    p = PCA().fit(tc.to_numpy())
    L = p.components_[:n_pcs].T
    # ChACRA sign convention: PC1 projection decreases with temperature
    proj = p.transform(tc.to_numpy())[:, 0]
    if np.polyfit(np.arange(len(proj)), proj, 1)[0] > 0:
        L[:, 0] *= -1
    return pd.DataFrame(L, index=tc.columns).reindex(list(contacts)).fillna(0.0).to_numpy()


def _run_mbar(args, project: Path, out_dir: Path):
    from chacra.reweight import (
        ResponseEstimator,
        cos_abs,
        get_occupancy_store,
        pc_frame,
    )

    store = get_occupancy_store(
        project, args.run, rebuild=args.rebuild_store,
        save_interval=args.save_interval,
    )
    print(f"[chacra response] store: {store.n_states} states x {store.n_frames} frames "
          f"x {store.n_contacts} contacts (source: {store.meta.get('contact_source')})")
    est = ResponseEstimator(store)
    print(f"[chacra response] REST2 decomposition residual: {est.fit_residual:.2e} "
          f"(should be ~1e-6; large values mean the Hamiltonian is not a + b*lam + c*sqrt(lam))")
    print(f"[chacra response] MBAR f_k (kT): {np.round(est.f_k, 1).tolist()}")

    lambdas = (
        np.asarray(args.lambdas, float) if args.lambdas else np.asarray(store.lambdas)
    )
    L = _pca_loadings(project, args.run, store.contacts)

    R1_rows, R2_rows, phi_rows, summary = [], [], [], []
    for lam in lambdas:
        r = est.at(lam)
        R1_rows.append(r.R1); R2_rows.append(r.R2); phi_rows.append(r.phi_mean)
        row = {"lambda": float(lam), "T_eff": float(store.temps[0] / lam), "ess": r.ess}
        # nearest sampled state → local ladder slope + single-state comparison
        k = int(np.argmin(np.abs(store.lambdas - lam)))
        if abs(store.lambdas[k] - lam) < 1e-9:
            fd = est.ladder_slope(k)
            row["state"] = k
            row["cos_R1_ladder_slope"] = cos_abs(r.R1, fd)
            row["slope_ratio_R1_over_ladder"] = float(r.R1 @ fd / (fd @ fd)) if fd @ fd > 0 else None
            if args.compare_single:
                s = est.single_state(k)
                row["cos_R1single_ladder_slope"] = cos_abs(s.R1, fd)
                row["cos_R1_R1single"] = cos_abs(r.R1, s.R1)
        if L is not None:
            row.update(pc_frame(r.R1, L))
        summary.append(row)
        print("  " + "  ".join(
            f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in row.items()
        ))

    idx = pd.Index(lambdas, name="lambda")
    pd.DataFrame(R1_rows, index=idx, columns=store.contacts).to_parquet(out_dir / "response_mbar_R1.parquet")
    pd.DataFrame(R2_rows, index=idx, columns=store.contacts).to_parquet(out_dir / "response_mbar_R2.parquet")
    pd.DataFrame(phi_rows, index=idx, columns=store.contacts).to_parquet(out_dir / "response_mbar_phi.parquet")

    # Headline: physical Hamiltonian (lambda = 1) with bootstrap errors
    head_lam = float(args.report_lambda)
    if args.n_boot > 0:
        r = est.bootstrap(head_lam, n_boot=args.n_boot, block=args.boot_block)
    else:
        r = est.at(head_lam)
    df = pd.DataFrame({"phi": r.phi_mean, "R1": r.R1, "R2": r.R2}, index=store.contacts)
    if r.R1_err is not None:
        df["R1_err"] = r.R1_err; df["R2_err"] = r.R2_err
    df.index.name = "contact"
    csv_path = out_dir / f"response_mbar_lambda{head_lam:.3f}.csv"
    df.to_csv(csv_path)
    print(f"\n[chacra response] lambda={head_lam:.3f} (T_eff={store.temps[0]/head_lam:.1f} K), "
          f"ESS={r.ess:.0f}; top {args.topk} contacts by |R1|:")
    top = df.reindex(df["R1"].abs().sort_values(ascending=False).index).head(args.topk)
    for name, row in top.iterrows():
        line = f"  {name:40s} phi={row['phi']:.3f}  R1={row['R1']:+.4f}"
        if "R1_err" in row:
            line += f" ± {row['R1_err']:.4f}"
        line += f"  R2={row['R2']:+.3f}"
        if "R2_err" in row:
            line += f" ± {row['R2_err']:.3f}"
        print(line)

    with open(out_dir / "response_mbar_summary.json", "w") as f:
        json.dump({
            "run": args.run, "estimator": "mbar",
            "n_states": store.n_states, "n_frames_per_state": store.n_frames,
            "n_contacts": store.n_contacts, "contact_source": store.meta.get("contact_source"),
            "decomposition_residual": est.fit_residual,
            "f_k": est.f_k.tolist(),
            "report_lambda": head_lam, "n_boot": args.n_boot,
            "per_lambda": summary,
        }, f, indent=2)
    print(f"[chacra response] wrote {csv_path}, response_mbar_{{R1,R2,phi}}.parquet, "
          f"response_mbar_summary.json in {out_dir}")


def _run_single(args, project: Path, out_dir: Path):
    from chacra.response import estimate_response_from_chacra_run, compare_to_pca

    result = estimate_response_from_chacra_run(
        project_dir=project, run=args.run, state_idx=args.state_idx,
        method=args.method, save_interval=args.save_interval,
    )
    if args.no_R2:
        result.R2 = None
    print(f"\n[chacra response] single-state estimate on {result.n_frames} frames "
          f"at state {result.state_idx} (β = {result.beta} in femto reduced units)")
    for k, v in result.diagnostics.items():
        print(f"  {k:25s}: {v}")
    df = result.to_dataframe()
    csv_path = out_dir / f"response_state_{result.state_idx}.csv"
    df.to_csv(csv_path)
    print(f"\n[chacra response] wrote {csv_path} ({len(df)} contacts)")
    print(f"\nTop {args.topk} contacts by |R^(1)|:")
    for name, row in result.topk(args.topk, by="R1").iterrows():
        line = f"  {name:40s}  R1={row['R1']:+.4e}"
        if "R2" in row:
            line += f"  R2={row['R2']:+.4e}"
        print(line)

    summary = {
        "run": args.run, "estimator": "single", "state_idx": result.state_idx,
        "method": args.method, "n_frames": result.n_frames, "beta": result.beta,
        "diagnostics": result.diagnostics, "n_contacts": len(result.contact_names),
    }
    if args.compare_pca:
        from chacra.ContactFrequencies import ContactFrequencies
        from chacra.run_config import RunConfig

        tc_path = project / "analysis_output" / f"run_{args.run}" / "total_contacts.parquet"
        if not tc_path.exists():
            sys.exit(f"  missing {tc_path} — run process-output first.")
        cfg = RunConfig(str(project / "chacra_run.json"))
        cf = ContactFrequencies(pd.read_parquet(tc_path), temps=np.round(np.array(cfg.get("temps"))), n_jobs=1)
        cmp_summary = compare_to_pca(result.R1, cf, result.contact_names)
        print(f"  |cos(PC1, R^(1))|  : {abs(cmp_summary['cos_pc1_R1']):.4f}  "
              "(PC1 is the ladder-AVERAGED slope; this is not a convergence target)")
        if cmp_summary.get("cos_pc2_R1") is not None:
            print(f"  |cos(PC2, R^(1))|  : {abs(cmp_summary['cos_pc2_R1']):.4f}")
        print(f"  aligned contacts   : {cmp_summary['n_aligned']} "
              f"(R1-only {cmp_summary['n_only_in_R1']}, PCA-only {cmp_summary['n_only_in_pca']})")
        summary["cos_pc1_R1"] = float(cmp_summary["cos_pc1_R1"])
        summary["cos_pc2_R1"] = (float(cmp_summary["cos_pc2_R1"])
                                 if cmp_summary.get("cos_pc2_R1") is not None else None)
        summary["n_aligned"] = cmp_summary["n_aligned"]
    with open(out_dir / f"response_state_{result.state_idx}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Contact response functions R^(1), R^(2) from a ChACRA HREMD run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_dir", type=str, default=".",
                        help="ChACRA project root (must contain chacra_run.json).")
    parser.add_argument("--run", type=int, default=None, help="HREMD run number. Default: latest.")
    parser.add_argument("--estimator", choices=["mbar", "single"], default="mbar",
                        help="mbar: pool all states with MBAR weights; single: one-replica cumulant.")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Contact-frame spacing (cycles) for THIS run, if it differs from chacra_run.json.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: analysis_output/run_N/.")
    parser.add_argument("--topk", type=int, default=20, help="How many top-|R1| contacts to print.")
    # mbar options
    g = parser.add_argument_group("mbar estimator")
    g.add_argument("--lambdas", type=float, nargs="+", default=None,
                   help="lambda values (beta_m/beta_0) at which to evaluate R1/R2. Default: the sampled states.")
    g.add_argument("--report_lambda", type=float, default=1.0,
                   help="lambda for the headline CSV with bootstrap errors (1.0 = physical Hamiltonian).")
    g.add_argument("--n_boot", type=int, default=50, help="Block-bootstrap resamples (0 to skip).")
    g.add_argument("--boot_block", type=int, default=50, help="Bootstrap block length in frames.")
    g.add_argument("--compare_single", action="store_true",
                   help="Also compute the single-state cumulant at each sampled state for comparison.")
    g.add_argument("--rebuild_store", action="store_true",
                   help="Rebuild contact_output/run_N/occupancy.npz even if cached.")
    # single-state options
    s = parser.add_argument_group("single-state estimator")
    s.add_argument("--state_idx", type=int, default=None, help="Reference replica. Default: ladder midpoint.")
    s.add_argument("--method", choices=["central", "forward", "backward"], default="central",
                   help="Finite-difference rule for dU/dk at the reference state.")
    s.add_argument("--no_R2", action="store_true", help="Skip R^(2).")
    s.add_argument("--compare_pca", action="store_true", help="Report |cos(PC1, R^(1))| (informational).")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    if not (project / "chacra_run.json").exists():
        sys.exit(f"chacra response: no chacra_run.json in {project}.")
    if args.run is None:
        args.run = _latest_run(project)
    out_dir = Path(args.output_dir) if args.output_dir else project / "analysis_output" / f"run_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[chacra response] project: {project}  run: {args.run}  estimator: {args.estimator}")

    if args.estimator == "mbar":
        _run_mbar(args, project, out_dir)
    else:
        _run_single(args, project, out_dir)


if __name__ == "__main__":
    main()
