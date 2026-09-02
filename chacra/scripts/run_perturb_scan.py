"""
chacra perturb-scan -- predicted contact-network response to per-residue
perturbations, from the HREMD ensemble (no new simulation).

    chacra perturb-scan --run 5                       # contact-basis chi, lambda = 1
    chacra perturb-scan --run 5 --generator energy \\
        --frame_stride 20 --platform CUDA             # OpenMM per-residue softening

Outputs (analysis_output/run_N/):
    perturb_<gen>_response.parquet   (contacts x residues)  R_{k,i}
    perturb_<gen>_coupling.parquet   (residues x residues)  A_{j,i}
    perturb_<gen>_ranking.csv        residues ranked by projected effect on the
                                     target contacts (default: top |PC1| loadings)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Perturbation scan: response of every contact to perturbing each residue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project_dir", default=".")
    p.add_argument("--run", type=int, default=None, help="Run number (default: latest with a store).")
    p.add_argument("--generator", choices=["contact", "energy"], default="contact",
                   help="contact: chi·J in the contact basis (cheap). energy: OpenMM dU/ds_i per residue.")
    p.add_argument("--lam", type=float, default=1.0,
                   help="lambda at which to evaluate the response (1.0 = physical Hamiltonian).")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Target contacts (names) or residues (chain:resn:resid) whose response is "
                        "scored. Default: contacts with the top --n_targets |PC1| loadings.")
    p.add_argument("--n_targets", type=int, default=20)
    p.add_argument("--include_self", action="store_true",
                   help="Keep a residue's own contacts in its score (default: allosteric only).")
    p.add_argument("--topk", type=int, default=20)
    g = p.add_argument_group("energy generator")
    g.add_argument("--system_file", default=None, help="Unmodified OpenMM system XML (default: chacra_run.json).")
    g.add_argument("--structure_file", default=None, help="Full-system PDB (default: chacra_run.json).")
    g.add_argument("--frame_stride", type=int, default=10, help="Use every n-th saved frame per state.")
    g.add_argument("--platform", default="CUDA")
    g.add_argument("--delta", type=float, default=0.02, help="Finite-difference step in s_i.")
    args = p.parse_args()

    from chacra.reweight import ResponseEstimator, get_occupancy_store
    from chacra.perturb import (
        contact_basis_scan, energy_basis_scan, energy_generators_for_store,
        rank_perturbation_sites, contact_residues,
    )
    from chacra.run_config import RunConfig

    project = Path(args.project_dir).resolve()
    cfg = RunConfig(str(project / "chacra_run.json")) if (project / "chacra_run.json").exists() else None
    if args.run is None:
        stores = sorted(project.glob("contact_output/run_*/occupancy.npz"),
                        key=lambda q: int(q.parent.name.split("_")[1]))
        if not stores:
            sys.exit("no contact_output/run_*/occupancy.npz; run process-output or `chacra response` first.")
        args.run = int(stores[-1].parent.name.split("_")[1])
    out_dir = project / "analysis_output" / f"run_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    store = get_occupancy_store(project, args.run)
    est = ResponseEstimator(store)
    print(f"[perturb-scan] run {args.run}: {store.n_states} states x {store.n_frames} frames x "
          f"{store.n_contacts} contacts; lambda={args.lam} ESS={est.ess(args.lam):.0f}")

    if args.generator == "contact":
        scan = contact_basis_scan(est, lam=args.lam)
    else:
        system_xml = args.system_file or (cfg.get("system_file") if cfg else None)
        structure = args.structure_file or (cfg.get("structure_file") if cfg else None)
        if not system_xml or not structure:
            sys.exit("--system_file and --structure_file are required for the energy generator.")
        idx, G, labels = energy_generators_for_store(
            store, project, args.run, str(project / system_xml), str(project / structure),
            frame_stride=args.frame_stride, platform=args.platform, delta=args.delta,
        )
        np.savez_compressed(out_dir / "perturb_energy_generators.npz", pooled_idx=idx, G=G,
                            residues=np.array(labels, dtype=object))
        scan = energy_basis_scan(est, idx, G, labels, lam=args.lam)
    print(f"[perturb-scan] generator={scan.generator}: response {scan.response.shape}, "
          f"effective samples {scan.ess:.0f}")

    # targets: default top-|PC1| contacts
    target_contacts, target_weights = None, None
    if args.targets:
        residues_given = [t for t in args.targets if t.count(":") == 2 and "-" not in t.split(":")[0]]
        if residues_given and len(residues_given) == len(args.targets):
            rs = set(args.targets)
            target_contacts = [c for c in scan.contacts if rs & set(contact_residues(c))]
        else:
            target_contacts = list(args.targets)
    else:
        tc_path = out_dir / "total_contacts.parquet"
        if tc_path.exists():
            from sklearn.decomposition import PCA

            tc = pd.read_parquet(tc_path)
            pc1 = pd.Series(PCA().fit(tc.to_numpy()).components_[0], index=tc.columns)
            target_contacts = pc1.abs().sort_values(ascending=False).head(args.n_targets).index.tolist()
            print(f"[perturb-scan] targets: top {len(target_contacts)} |PC1| contacts")
        else:
            print("[perturb-scan] no total_contacts.parquet; scoring against ALL contacts.")

    ranking = rank_perturbation_sites(scan, target_contacts, target_weights,
                                      exclude_self=not args.include_self)
    tag = scan.generator
    scan.response_df().to_parquet(out_dir / f"perturb_{tag}_response.parquet")
    scan.coupling_df().to_parquet(out_dir / f"perturb_{tag}_coupling.parquet")
    ranking.to_csv(out_dir / f"perturb_{tag}_ranking.csv", index=False)
    with open(out_dir / f"perturb_{tag}_summary.json", "w") as f:
        json.dump({"run": args.run, "generator": tag, "lambda": scan.lam, "ess": scan.ess,
                   "n_targets": len(target_contacts) if target_contacts else None,
                   "targets": target_contacts, "exclude_self": not args.include_self}, f, indent=2)
    print(f"\nTop {args.topk} perturbation sites (projected response on targets, "
          f"{'incl.' if args.include_self else 'excl.'} own contacts):")
    print(ranking.head(args.topk).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n[perturb-scan] wrote perturb_{tag}_{{response,coupling}}.parquet, "
          f"perturb_{tag}_ranking.csv in {out_dir}")


if __name__ == "__main__":
    main()
