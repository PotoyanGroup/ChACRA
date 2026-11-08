"""
chacra response -- contact response functions R^(1), R^(2) from finished runs.

Two estimators on the same occupancy store (per-state per-frame contacts plus
femto's reduced potentials under every state):

* ``--estimator mbar`` (default): pools the frames of ALL states with MBAR
  weights and evaluates d<phi>/dlambda, d2<phi>/dlambda2 at any lambda,
  including lambda = 1, the physical (unscaled) Hamiltonian.
* ``--estimator single``: the single-state cumulant at each sampled state
  (only that state's frames, uniform weights, the same analytic REST2
  generators), the one-replica limit of the same identity.

``--runs 4-7`` pools several runs of one ladder before either estimate.

Each estimate is scored against the LOCAL finite-difference slope of the
frequency ladder at the same lambda (the quantity the estimators target) and
placed in the PC1-PC3 frame of ChACRA's PCA.  PC1 is the ladder-averaged slope,
not the local one, so |cos(PC1, R1)| is reported but is not a convergence
criterion.
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


def _describe_store(store, runs):
    src = store.meta.get("contact_source")
    where = f"runs {runs}" if len(runs) > 1 else f"run {runs[0]}"
    print(f"[chacra response] store ({where}): {store.n_states} states x {store.n_frames} frames "
          f"x {store.n_contacts} contacts (source: {src}; u at {store.meta.get('u_temperature')} K, "
          f"scored at {store.meta.get('u_source_temperature')} K"
          f"{'; legacy store rescaled' if store.meta.get('legacy_rescaled') else ''})")


def _print_row(row):
    print("  " + "  ".join(
        f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in row.items()
    ))


def _print_top(df, topk, label):
    print(f"\n[chacra response] {label}; top {topk} contacts by |R1|:")
    top = df.reindex(df["R1"].abs().sort_values(ascending=False).index).head(topk)
    for name, row in top.iterrows():
        line = f"  {name:40s} phi={row['phi']:.3f}  R1={row['R1']:+.4f}"
        if "R1_err" in row:
            line += f" ± {row['R1_err']:.4f}"
        line += f"  R2={row['R2']:+.3f}"
        if "R2_err" in row:
            line += f" ± {row['R2_err']:.3f}"
        print(line)


def _run_mbar(args, project: Path, out_dir: Path, store, runs, tag):
    from chacra.reweight import ResponseEstimator, cos_abs, pc_frame

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
        # nearest sampled state -> local ladder slope + single-state comparison
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
                row["cos_R2_R2single"] = cos_abs(r.R2, s.R2)
        if L is not None:
            row.update(pc_frame(r.R1, L))
        summary.append(row)
        _print_row(row)

    idx = pd.Index(lambdas, name="lambda")
    pd.DataFrame(R1_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_mbar_R1{tag}.parquet")
    pd.DataFrame(R2_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_mbar_R2{tag}.parquet")
    pd.DataFrame(phi_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_mbar_phi{tag}.parquet")

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
    csv_path = out_dir / f"response_mbar_lambda{head_lam:.3f}{tag}.csv"
    df.to_csv(csv_path)
    _print_top(df, args.topk,
               f"lambda={head_lam:.3f} (T_eff={store.temps[0]/head_lam:.1f} K), ESS={r.ess:.0f}")

    with open(out_dir / f"response_mbar_summary{tag}.json", "w") as f:
        json.dump({
            "run": args.run, "runs": runs, "estimator": "mbar",
            "n_states": store.n_states, "n_frames_per_state": store.n_frames,
            "n_contacts": store.n_contacts, "contact_source": store.meta.get("contact_source"),
            "decomposition_residual": est.fit_residual,
            "u_temperature": store.meta.get("u_temperature"),
            "u_source_temperature": store.meta.get("u_source_temperature"),
            "legacy_rescaled": bool(store.meta.get("legacy_rescaled", False)),
            "segments": store.meta.get("segments"),
            "f_k": est.f_k.tolist(),
            "report_lambda": head_lam, "n_boot": args.n_boot,
            "per_lambda": summary,
        }, f, indent=2)
    print(f"[chacra response] wrote {csv_path.name}, response_mbar_{{R1,R2,phi}}{tag}.parquet, "
          f"response_mbar_summary{tag}.json in {out_dir}")


def _run_single(args, project: Path, out_dir: Path, store, runs, tag):
    from chacra.reweight import (
        ResponseEstimator,
        cos_abs,
        ladder_slope,
        pc_frame,
        single_state_response,
    )
    from chacra.response import compare_to_pca

    states = [args.state_idx] if args.state_idx is not None else list(range(store.n_states))
    for k in states:
        if not 0 <= k < store.n_states:
            sys.exit(f"--state_idx {k} out of range [0, {store.n_states - 1}]")
    head_k = int(np.argmin(np.abs(store.lambdas - float(args.report_lambda))))
    if head_k not in states:
        head_k = states[0]
    est_mbar = ResponseEstimator(store) if args.compare_mbar else None
    L = _pca_loadings(project, args.run, store.contacts)

    R1_rows, R2_rows, phi_rows, summary, head = [], [], [], [], None
    for k in states:
        n_boot = args.n_boot if k == head_k else 0
        r = single_state_response(store, k, n_boot=n_boot, block=args.boot_block)
        if k == head_k:
            head = r
        R1_rows.append(r.R1); R2_rows.append(r.R2); phi_rows.append(r.phi_mean)
        fd = ladder_slope(store, k)
        row = {"lambda": r.lam, "T_eff": float(store.temps[k]), "state": k,
               "n_frames": int(store.n_frames),
               "cos_R1_ladder_slope": cos_abs(r.R1, fd),
               "slope_ratio_R1_over_ladder": float(r.R1 @ fd / (fd @ fd)) if fd @ fd > 0 else None}
        if est_mbar is not None:
            m = est_mbar.at(r.lam)
            row["cos_R1single_R1mbar"] = cos_abs(r.R1, m.R1)
            row["cos_R2single_R2mbar"] = cos_abs(r.R2, m.R2)
            row["R1_norm_ratio_single_over_mbar"] = (
                float(np.linalg.norm(r.R1) / np.linalg.norm(m.R1)) if np.linalg.norm(m.R1) > 0 else None
            )
        if L is not None:
            row.update(pc_frame(r.R1, L))
        summary.append(row)
        _print_row(row)

    idx = pd.Index([store.lambdas[k] for k in states], name="lambda")
    pd.DataFrame(R1_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_single_R1{tag}.parquet")
    pd.DataFrame(R2_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_single_R2{tag}.parquet")
    pd.DataFrame(phi_rows, index=idx, columns=store.contacts).to_parquet(out_dir / f"response_single_phi{tag}.parquet")

    df = pd.DataFrame({"phi": head.phi_mean, "R1": head.R1, "R2": head.R2}, index=store.contacts)
    if head.R1_err is not None:
        df["R1_err"] = head.R1_err; df["R2_err"] = head.R2_err
    df.index.name = "contact"
    csv_path = out_dir / f"response_single_state{head_k}{tag}.csv"
    df.to_csv(csv_path)
    _print_top(df, args.topk,
               f"single-state estimate at state {head_k} (lambda={head.lam:.3f}, "
               f"T_eff={store.temps[head_k]:.1f} K, {store.n_frames} frames)")

    out = {
        "run": args.run, "runs": runs, "estimator": "single", "states": states,
        "headline_state": head_k, "n_frames_per_state": int(store.n_frames),
        "n_contacts": store.n_contacts, "contact_source": store.meta.get("contact_source"),
        "u_temperature": store.meta.get("u_temperature"),
        "u_source_temperature": store.meta.get("u_source_temperature"),
        "legacy_rescaled": bool(store.meta.get("legacy_rescaled", False)),
        "segments": store.meta.get("segments"),
        "n_boot": args.n_boot, "per_state": summary,
    }
    if args.compare_pca:
        from chacra.ContactFrequencies import ContactFrequencies
        from chacra.run_config import RunConfig

        tc_path = project / "analysis_output" / f"run_{args.run}" / "total_contacts.parquet"
        if not tc_path.exists():
            sys.exit(f"  missing {tc_path} -- run process-output first.")
        cfg = RunConfig(str(project / "chacra_run.json"))
        cf = ContactFrequencies(pd.read_parquet(tc_path), temps=np.round(np.array(cfg.get("temps"))), n_jobs=1)
        cmp_summary = compare_to_pca(head.R1, cf, store.contacts)
        print(f"  |cos(PC1, R^(1))|  : {abs(cmp_summary['cos_pc1_R1']):.4f}  "
              "(PC1 is the ladder-AVERAGED slope; this is not a convergence target)")
        if cmp_summary.get("cos_pc2_R1") is not None:
            print(f"  |cos(PC2, R^(1))|  : {abs(cmp_summary['cos_pc2_R1']):.4f}")
        out["cos_pc1_R1"] = float(cmp_summary["cos_pc1_R1"])
        out["cos_pc2_R1"] = (float(cmp_summary["cos_pc2_R1"])
                             if cmp_summary.get("cos_pc2_R1") is not None else None)
        out["n_aligned"] = cmp_summary["n_aligned"]
    with open(out_dir / f"response_single_summary{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[chacra response] wrote {csv_path.name}, response_single_{{R1,R2,phi}}{tag}.parquet, "
          f"response_single_summary{tag}.json in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Contact response functions R^(1), R^(2) from ChACRA HREMD runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_dir", type=str, default=".",
                        help="ChACRA project root (must contain chacra_run.json).")
    parser.add_argument("--run", type=int, default=None, help="HREMD run number. Default: latest.")
    parser.add_argument("--runs", type=str, default=None,
                        help="Pool several runs of the same ladder before estimating, e.g. '4-7', "
                             "'4,6,7' or 'all'. Outputs go to analysis_output/run_<last>/ tagged _runs<a>-<b>.")
    parser.add_argument("--estimator", choices=["mbar", "single"], default="mbar",
                        help="mbar: pool all states with MBAR weights; single: one-state cumulant at each sampled state.")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Contact-frame spacing (cycles) for THIS run, if it differs from chacra_run.json "
                             "(runs written since 2026-11 carry their own in run_meta.json).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: analysis_output/run_N/.")
    parser.add_argument("--topk", type=int, default=20, help="How many top-|R1| contacts to print.")
    parser.add_argument("--report_lambda", type=float, default=1.0,
                        help="lambda for the headline CSV with bootstrap errors (1.0 = physical Hamiltonian; "
                             "for the single estimator the nearest sampled state).")
    parser.add_argument("--n_boot", type=int, default=50, help="Block-bootstrap resamples (0 to skip).")
    parser.add_argument("--boot_block", type=int, default=50, help="Bootstrap block length in frames.")
    parser.add_argument("--rebuild_store", action="store_true",
                        help="Rebuild contact_output/run_N/occupancy.npz even if cached.")
    parser.add_argument("--u_source_temperature", type=float, default=None,
                        help="Temperature (K) femto scored the stored reduced potentials at. Default: the "
                             "run's run_meta.json, then chacra_run.json 'hremd_temperature', else femto's "
                             "default 298.15 K (every ChACRA run before 2026-11). Energies are rescaled to "
                             "beta at temps[0].")
    g = parser.add_argument_group("mbar estimator")
    g.add_argument("--lambdas", type=float, nargs="+", default=None,
                   help="lambda values (beta_m/beta_0) at which to evaluate R1/R2. Default: the sampled states.")
    g.add_argument("--compare_single", action="store_true",
                   help="Also compute the single-state cumulant at each sampled state for comparison.")
    s = parser.add_argument_group("single-state estimator")
    s.add_argument("--state_idx", type=int, default=None,
                   help="Evaluate one state only. Default: every sampled state.")
    s.add_argument("--compare_mbar", action="store_true",
                   help="Also solve MBAR and report cos(R1_single, R1_mbar) at each state.")
    s.add_argument("--compare_pca", action="store_true", help="Report |cos(PC1, R^(1))| (informational).")
    s.add_argument("--method", type=str, default=None, help=argparse.SUPPRESS)  # retired
    s.add_argument("--no_R2", action="store_true", help=argparse.SUPPRESS)      # retired
    args = parser.parse_args()

    from chacra.reweight import LEGACY_U_TEMPERATURE, get_pooled_store, parse_runs

    if args.method is not None:
        print("[chacra response] --method is retired: du/dlambda is the analytic REST2 generator.")
    project = Path(args.project_dir).resolve()
    if not (project / "chacra_run.json").exists():
        sys.exit(f"chacra response: no chacra_run.json in {project}.")
    runs = parse_runs(args.runs, project) if args.runs else []
    if runs:
        args.run = runs[-1]
    elif args.run is None:
        args.run = _latest_run(project)
    if not runs:
        runs = [args.run]
    tag = f"_runs{runs[0]}-{runs[-1]}" if len(runs) > 1 else ""
    out_dir = Path(args.output_dir) if args.output_dir else project / "analysis_output" / f"run_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[chacra response] project: {project}  runs: {runs}  estimator: {args.estimator}")

    src_T = args.u_source_temperature
    store = get_pooled_store(
        project, runs, rebuild=args.rebuild_store,
        legacy_u_temperature=src_T if src_T is not None else LEGACY_U_TEMPERATURE,
        save_interval=args.save_interval, u_source_temperature=src_T,
    )
    _describe_store(store, runs)

    if args.estimator == "mbar":
        _run_mbar(args, project, out_dir, store, runs, tag)
    else:
        _run_single(args, project, out_dir, store, runs, tag)


if __name__ == "__main__":
    main()
