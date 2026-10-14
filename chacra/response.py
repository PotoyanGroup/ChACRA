"""
Single-state response estimators for ChACRA contact frequencies.

Background
----------
The "ChACRA identity" of the AAlostery Paper 1 framework states that for any
1D Hamiltonian perturbation U(lambda) = U_0 + lambda * dU around a reference
state lambda_0, the first- and second-order responses of contact occupancies
phi_k are connected cumulants of equilibrium fluctuations at lambda_0:

    R^{(1)}_k = d<phi_k>/dlambda |_{lambda_0}    = -beta * Cov(phi_k, dU)
    R^{(2)}_k = d^2<phi_k>/dlambda^2 |_{lambda_0} = beta^2 * <delta phi_k * (delta dU)^2>_c

Standard ChACRA estimates these by running HREMD across many lambda values
and reading the slope from the freqs(lambda) curve. The cumulant form
above is mathematically equivalent at leading order, but estimable from the
ensemble at a single replica — which (in principle) collapses the cost of
the full lambda ladder.

This module implements the cumulant estimators and provides loaders that
turn the femto + ultracontacts output of `chacra run-hremd` into the
required per-frame arrays. A `compare_to_pca` helper lets you check that
the single-state R^{(1)} reproduces ChACRA's PC1 to high cosine.

Conventions
-----------
- `phi`: per-frame contact occupancies, shape (n_frames, n_contacts), values
  in [0, 1] (typical ultracontacts output is binary 0/1 per frame).
- `dU`: per-frame value of the perturbation generator dU/dlambda evaluated
  at the reference state, shape (n_frames,). Units: kJ/mol or kcal/mol —
  must match `beta` so that `beta * dU` is dimensionless.
- `beta`: 1/(k_B T) at the reference state, in inverse energy units that
  cancel `dU`.
- Returned `R1`, `R2`: same length as the contact axis, in units of
  d<phi>/dlambda — dimensionless per unit lambda.

Sign convention follows AAlostery `test_b_from_freqs.py`: lambda runs from
0 (low effective T = unscaled) to 1 (high effective T = max scaling), so
positive `R1_k` means contact k *gains* probability as the perturbation
strengthens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Core cumulant estimators                                                     #
# --------------------------------------------------------------------------- #


def cumulant_R1(
    phi: np.ndarray,
    dU: np.ndarray,
    beta: float,
) -> np.ndarray:
    """First-order ChACRA response from a single equilibrium ensemble.

    Implements R^{(1)}_k = -beta * Cov(phi_k, dU).

    Parameters
    ----------
    phi : (n_frames, n_contacts) array
        Per-frame contact occupancies at the reference state.
    dU : (n_frames,) array
        Per-frame value of the perturbation generator at the reference state.
    beta : float
        Inverse temperature, in units that cancel `dU`.

    Returns
    -------
    R1 : (n_contacts,) array
    """
    phi = np.asarray(phi, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    if phi.shape[0] != dU.shape[0]:
        raise ValueError(
            f"phi has {phi.shape[0]} frames, dU has {dU.shape[0]}; must match."
        )
    n = phi.shape[0]
    # Use the unbiased covariance (1/(n-1)); same as np.cov(..., bias=False)
    phi_c = phi - phi.mean(axis=0, keepdims=True)
    dU_c = dU - dU.mean()
    cov = (phi_c * dU_c[:, None]).sum(axis=0) / (n - 1)
    return -beta * cov


def cumulant_R2(
    phi: np.ndarray,
    dU: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Second-order ChACRA response, assuming linear-in-lambda perturbation.

    Implements R^{(2)}_k = beta^2 * <delta phi_k . (delta dU)^2>_c, where
    the connected third moment is
        <abc>_c = <abc> - <a><bc> - <b><ac> - <c><ab> + 2<a><b><c>.

    For a linear-in-lambda Hamiltonian U(lambda) = U_0 + lambda * dU there is
    no d^2 U / dlambda^2 contribution. If the underlying scaling is non-linear
    (e.g. pure REST2 with sqrt(lambda) on protein-solvent), this estimator is
    only accurate to O(lambda - lambda_0).

    Returns
    -------
    R2 : (n_contacts,) array
        Same length as the contact axis. Units: d^2<phi>/dlambda^2.
    """
    phi = np.asarray(phi, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    n = phi.shape[0]

    phi_bar = phi.mean(axis=0)
    dU_bar = dU.mean()
    dU_c = dU - dU_bar
    phi_c = phi - phi_bar  # (n, K)

    # E[phi * dU^2]
    E_phi_dU2 = (phi * (dU ** 2)[:, None]).sum(axis=0) / n
    # E[phi * dU]
    E_phi_dU = (phi * dU[:, None]).sum(axis=0) / n
    # E[dU^2]
    E_dU2 = (dU ** 2).mean()
    # E[dU]
    E_dU = dU_bar

    # Connected third moment <delta phi_k * (delta dU)^2>
    # = E[phi_k dU^2] - phi_bar_k E[dU^2] - 2 E[dU] (E[phi_k dU] - phi_bar_k E[dU])
    third_c = (
        E_phi_dU2
        - phi_bar * E_dU2
        - 2.0 * E_dU * (E_phi_dU - phi_bar * E_dU)
    )
    return (beta ** 2) * third_c


def cos_align(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity, matching AAlostery's convention."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(abs(np.dot(a, b) / (na * nb)))


# --------------------------------------------------------------------------- #
# femto / ultracontacts loaders                                                #
# --------------------------------------------------------------------------- #


def load_perturbation_energy_femto(
    samples_arrow: str | os.PathLike,
    state_idx: int,
    save_interval: int = 1,
    method: str = "central",
) -> tuple[np.ndarray, float]:
    """Read the per-cycle perturbation energy dU at one reference state.

    femto's `samples.arrow` stores `u_kn[cycle][k][n]`, the reduced energy
    of the configuration that was sampled at state n on cycle `cycle`,
    re-evaluated under the Hamiltonian of state k. This function picks the
    column for `state_idx` (i.e. the configurations sampled at that state),
    and uses finite differences in `k` to estimate dU/dlambda at the same
    state.

    Parameters
    ----------
    samples_arrow : path
        femto HREMD samples (`replica_trajectories/run_N/samples.arrow`).
    state_idx : int
        Reference replica / state index (0 = lowest effective T = unscaled).
    save_interval : int
        Cycle stride for trajectory writes. The returned energy array is
        sliced with this stride so it aligns frame-for-frame with the
        per-frame contact data computed from `state_trajectories/`.
    method : {"central", "forward", "backward"}
        Finite-difference rule for dU/dk in state index. "central" (default)
        requires `state_idx` to be an interior state; "forward"/"backward"
        for the boundaries.

    Returns
    -------
    dU : (n_frames,) array of perturbation energies (femto's reduced units)
    delta_lambda : float
        Spacing in lambda between adjacent states used in the finite
        difference. Returned so the caller can rescale dU if lambda is not
        the natural step (for example if comparing to T_eff-spaced ladders).
    """
    from chacra.trajectories.process_hremd import load_femto_data

    df = load_femto_data(str(samples_arrow))
    n_states = df["u_kn"][0].shape[0]
    if not 0 <= state_idx < n_states:
        raise ValueError(f"state_idx={state_idx} out of range [0, {n_states-1}]")

    # femto convention: u_kn[k, m] is replica m's config under Hamiltonian k,
    # and replica_to_state_idx[r] gives the state of replica r. To extract the
    # configuration currently at `state_idx` we must locate the replica r
    # such that replica_to_state_idx[r] == state_idx — this changes per cycle
    # because exchanges shuffle the (replica → state) assignment.
    n_cycles = len(df)
    u_per_cycle = np.empty((n_cycles, n_states), dtype=np.float64)
    for cycle, (u_row, r2s) in enumerate(
        zip(df["u_kn"], df["replica_to_state_idx"])
    ):
        U = np.vstack(u_row)  # (n_states, n_states): [Hamiltonian k, replica m]
        r2s_arr = np.asarray(r2s)
        matches = np.where(r2s_arr == state_idx)[0]
        if matches.size == 0:
            raise RuntimeError(
                f"No replica at state {state_idx} on cycle {cycle}; "
                f"replica_to_state_idx={r2s_arr.tolist()}"
            )
        replica_at_ref = int(matches[0])
        u_per_cycle[cycle] = U[:, replica_at_ref]

    # Finite difference in k at the reference state.
    if method == "central":
        if state_idx in (0, n_states - 1):
            raise ValueError(
                "method='central' requires an interior state. "
                "Use method='forward' or 'backward' at the ladder ends."
            )
        dU_per_cycle = 0.5 * (u_per_cycle[:, state_idx + 1]
                              - u_per_cycle[:, state_idx - 1])
        delta_k = 1.0  # in units of state-index steps
    elif method == "forward":
        if state_idx == n_states - 1:
            raise ValueError("method='forward' invalid at top of ladder.")
        dU_per_cycle = u_per_cycle[:, state_idx + 1] - u_per_cycle[:, state_idx]
        delta_k = 1.0
    elif method == "backward":
        if state_idx == 0:
            raise ValueError("method='backward' invalid at bottom of ladder.")
        dU_per_cycle = u_per_cycle[:, state_idx] - u_per_cycle[:, state_idx - 1]
        delta_k = 1.0
    else:
        raise ValueError(f"unknown method={method!r}")

    # Subsample to the trajectory save cadence so dU aligns with phi.  femto
    # saves frames at absolute cycle % save_interval == 0, so the first saved
    # row of a continuation run is generally not row 0.
    from chacra.trajectories.process_hremd import saved_frame_rows

    rows = saved_frame_rows(df, save_interval)
    dU = dU_per_cycle[rows]
    return dU, float(delta_k)


def load_per_frame_contacts(
    parquet_path: str | os.PathLike,
    n_frames: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load per-frame contact occupancies from an ultracontacts parquet.

    ultracontacts writes per-frame data in *long event-log* format:
    one row per (frame, interaction-type, atom_pair) event::

        frame  itype   atom1            atom2
        0      sb      B:ASP:146:OD1    B:ARG:131:NH2
        0      sb      A:ASP:118:OD1    A:LYS:136:NZ
        ...

    We canonicalise atoms → residues (``chain:resname:resid``), collapse
    interaction types onto a residue pair (matching the freqs convention),
    and pivot to a (n_frames, n_contacts) binary occupancy matrix.

    Wide-format parquets (rows=frames, columns=contact identifiers, cells
    in {0,1}) are also accepted as a fallback for non-ultracontacts inputs.

    Returns
    -------
    phi : (n_frames, n_contacts) float64 array (binary 0/1)
    contact_names : list[str] of length n_contacts, formatted "res1--res2"
    """
    df = pd.read_parquet(str(parquet_path))
    # Long-event schema produced by `ultracontacts contacts`
    if {"frame", "atom1", "atom2"}.issubset(df.columns):
        nf = int(n_frames if n_frames is not None else df["frame"].max() + 1)
        # Truncate atom strings (chain:resname:resid:atom) to residue identity
        # (chain:resname:resid) so the contact key matches getcontacts /
        # ChACRA freqs conventions.
        a = df["atom1"].str.split(":").str[:3].str.join(":")
        b = df["atom2"].str.split(":").str[:3].str.join(":")
        first = np.where(a <= b, a, b)
        second = np.where(a <= b, b, a)
        pair = pd.Series([f"{x}-{y}" for x, y in zip(first, second)])
        contacts = sorted(pair.unique())
        contact_idx = {c: i for i, c in enumerate(contacts)}
        phi = np.zeros((nf, len(contacts)), dtype=np.float64)
        cols = pair.map(contact_idx).to_numpy()
        rows = df["frame"].to_numpy()
        # Multiple events on the same (frame, residue-pair) collapse to a
        # single occupancy of 1 (any-itype indicator).
        phi[rows, cols] = 1.0
        return phi, contacts
    # Wide-format fallback
    contact_names = [str(c) for c in df.columns]
    phi = df.to_numpy(dtype=np.float64, copy=False)
    return phi, contact_names


# --------------------------------------------------------------------------- #
# getcontacts (legacy GROMACS HREMD) loader                                   #
# --------------------------------------------------------------------------- #


def load_per_frame_contacts_getcontacts(
    contacts_tsv: str | os.PathLike,
    n_frames: int | None = None,
    interaction_types: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load per-frame occupancies from a getcontacts per-frame TSV.

    Format (whitespace-separated):
        # total_frames:NN  interaction_types:...
        frame  itype  res1  res2  [extras]

    A pair (res1, res2) is "in contact" on a given frame if any of the
    selected interaction types fires. This is the same convention used by
    `get-contact-frequencies` to produce the freqs files.

    Parameters
    ----------
    contacts_tsv : path
        getcontacts per-frame contact file.
    n_frames : int, optional
        Total number of MD frames. If None, parsed from the TSV header.
    interaction_types : sequence of str, optional
        Restrict to these interaction types (e.g. ["sb", "vdw", "hb"]).
        Default: all types present.

    Returns
    -------
    phi : (n_frames, n_contacts) bool→float64 array
    contact_names : list[str], formatted as "res1--res2" (canonicalized)
    """
    contacts_tsv = str(contacts_tsv)
    selected = set(interaction_types) if interaction_types is not None else None

    # Parse header for n_frames if missing.
    if n_frames is None:
        with open(contacts_tsv) as f:
            for line in f:
                if line.startswith("#") and "total_frames" in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith("total_frames"):
                            n_frames = int(p.split(":")[1])
                            break
                    break
        if n_frames is None:
            raise ValueError(
                f"Could not parse total_frames from header of {contacts_tsv}"
            )

    # Stream rows once to collect contact pairs and frame indices.
    pair_to_idx: dict[tuple[str, str], int] = {}
    rows: list[tuple[int, int]] = []  # (frame_idx, contact_idx)
    with open(contacts_tsv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                frame = int(parts[0])
            except ValueError:
                continue
            itype = parts[1]
            if selected is not None and itype not in selected:
                continue
            # Truncate atom strings (chain:resname:resid:atom) to residue
            # identity (chain:resname:resid) so the contact key matches the
            # convention used by getcontacts freqs files and ChACRA PCA.
            r1 = ":".join(parts[2].split(":")[:3])
            r2 = ":".join(parts[3].split(":")[:3])
            key = (r1, r2) if r1 <= r2 else (r2, r1)
            idx = pair_to_idx.get(key)
            if idx is None:
                idx = len(pair_to_idx)
                pair_to_idx[key] = idx
            rows.append((frame, idx))

    n_contacts = len(pair_to_idx)
    phi = np.zeros((n_frames, n_contacts), dtype=np.float64)
    if rows:
        rows_arr = np.asarray(rows, dtype=np.int64)
        # Multiple interaction types on the same frame collapse to a single
        # "is in contact" indicator (max), matching the freqs convention.
        phi[rows_arr[:, 0], rows_arr[:, 1]] = 1.0

    contact_names = [f"{a}-{b}" for (a, b) in pair_to_idx]
    return phi, contact_names


# --------------------------------------------------------------------------- #
# Bundled estimator + comparison helpers                                       #
# --------------------------------------------------------------------------- #


@dataclass
class ResponseResult:
    """Container for the single-state response estimate."""

    R1: np.ndarray
    R2: np.ndarray | None
    contact_names: list[str]
    beta: float
    state_idx: int
    n_frames: int
    diagnostics: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"R1": self.R1}, index=self.contact_names)
        if self.R2 is not None:
            df["R2"] = self.R2
        df.index.name = "contact"
        return df

    def topk(self, k: int = 20, by: str = "R1") -> pd.DataFrame:
        df = self.to_dataframe()
        if by not in df.columns:
            raise KeyError(f"no column {by!r} (have {df.columns.tolist()})")
        return df.reindex(df[by].abs().sort_values(ascending=False).index).head(k)


def estimate_response(
    phi: np.ndarray,
    dU: np.ndarray,
    beta: float,
    contact_names: Sequence[str] | None = None,
    state_idx: int = -1,
    compute_R2: bool = True,
) -> ResponseResult:
    """Compute R^{(1)} (and optionally R^{(2)}) from per-frame data."""
    phi = np.asarray(phi, dtype=np.float64)
    dU = np.asarray(dU, dtype=np.float64)
    if contact_names is None:
        contact_names = [f"contact_{i}" for i in range(phi.shape[1])]
    R1 = cumulant_R1(phi, dU, beta)
    R2 = cumulant_R2(phi, dU, beta) if compute_R2 else None

    diagnostics = {
        "phi_mean": float(phi.mean()),
        "phi_active_fraction": float((phi.mean(axis=0) > 0).mean()),
        "dU_mean": float(dU.mean()),
        "dU_std": float(dU.std()),
        "n_frames": int(phi.shape[0]),
        "n_contacts": int(phi.shape[1]),
    }
    return ResponseResult(
        R1=R1,
        R2=R2,
        contact_names=list(contact_names),
        beta=beta,
        state_idx=state_idx,
        n_frames=int(phi.shape[0]),
        diagnostics=diagnostics,
    )


def compare_to_pca(
    R1: np.ndarray,
    cf,  # ContactFrequencies
    contact_names: Sequence[str],
) -> dict:
    """Compute |cos(PC1, R^{(1)})| and friends, matching test_b_from_freqs.

    `cf` is a `chacra.ContactFrequencies.ContactFrequencies` instance whose
    `cf.cpca.loadings` (or equivalent attribute) provides the PC1 loadings
    over its own contact ordering. We align by name; contacts present in
    only one of the two are dropped from the comparison and reported.

    Returns
    -------
    dict with keys: cos_pc1_R1, cos_pc2_R1, n_aligned, n_only_in_R1,
    n_only_in_pca, pc1_aligned, R1_aligned, contact_names_aligned.
    """
    # Pull PC1, PC2 loadings + contact order out of cf.
    # Must use SIGNED loadings: `norm_loadings` is per-PC |loading|/max
    # (sign-stripped), which structurally depresses the cosine — a contact
    # that melts in and a contact that melts out both count as positive.
    cpca = cf.cpca
    if hasattr(cpca, "loadings"):
        loadings = np.asarray(cpca.loadings)
    elif hasattr(cpca, "original_loadings"):
        loadings = np.asarray(cpca.original_loadings)
    elif hasattr(cpca, "norm_loadings"):
        import warnings

        warnings.warn(
            "compare_to_pca: only sign-stripped `norm_loadings` available; "
            "cos_pc1_R1 will be structurally depressed.",
            stacklevel=2,
        )
        loadings = np.asarray(cpca.norm_loadings)
    else:
        raise AttributeError(
            "Could not find PC loadings on cf.cpca "
            "(checked: norm_loadings, original_loadings, loadings)."
        )

    pca_contacts = list(cf.freqs.columns)
    # ChACRA stores loadings as (n_contacts, n_components) — opposite of the
    # sklearn / SVD convention. PC_k vector across contacts is column k.
    if loadings.shape[0] == len(pca_contacts):
        pc1 = loadings[:, 0]
        pc2 = loadings[:, 1] if loadings.shape[1] >= 2 else None
    else:
        pc1 = loadings[0]
        pc2 = loadings[1] if loadings.shape[0] >= 2 else None

    # Align by contact name (intersection).
    R1_dict = dict(zip(contact_names, R1))
    pc1_dict = dict(zip(pca_contacts, pc1))
    pc2_dict = dict(zip(pca_contacts, pc2)) if pc2 is not None else None

    # Union alignment: a contact that never forms at the reference state has
    # phi == 0 on every frame and therefore R1 == 0 exactly; dropping it (the
    # old intersection behaviour) silently removes PCA-loaded contacts from
    # the comparison.  Contacts absent from the PCA get loading 0.
    seen = set()
    aligned_names = [
        c for c in list(contact_names) + list(pca_contacts)
        if not (c in seen or seen.add(c))
    ]
    R1_a = np.array([R1_dict.get(c, 0.0) for c in aligned_names])
    pc1_a = np.array([pc1_dict.get(c, 0.0) for c in aligned_names])
    pc2_a = (
        np.array([pc2_dict.get(c, 0.0) for c in aligned_names]) if pc2_dict else None
    )

    only_in_R1 = sum(1 for c in contact_names if c not in pc1_dict)
    only_in_pca = sum(1 for c in pca_contacts if c not in R1_dict)

    out = {
        "cos_pc1_R1": cos_align(pc1_a, R1_a),
        "cos_pc2_R1": cos_align(pc2_a, R1_a) if pc2_a is not None else None,
        "n_aligned": len(aligned_names),
        "n_only_in_R1": only_in_R1,
        "n_only_in_pca": only_in_pca,
        "pc1_aligned": pc1_a,
        "R1_aligned": R1_a,
        "contact_names_aligned": aligned_names,
    }
    return out


# --------------------------------------------------------------------------- #
# End-to-end: estimate_response_from_chacra_run                                #
# --------------------------------------------------------------------------- #


def estimate_response_from_chacra_run(
    project_dir: str | os.PathLike,
    run: int = 1,
    state_idx: int | None = None,
    method: str = "central",
    save_interval: int | None = None,
) -> ResponseResult:
    """Run the full single-state estimator on a ChACRA project directory.

    Expects the standard ChACRA layout:
        project_dir/
            chacra_run.json
            replica_trajectories/run_N/samples.arrow
            contact_output/run_N/contacts/cont_state_{i}.parquet

    Picks the middle replica by default (best central-difference accuracy
    and best chance of being well-equilibrated under either side).

    Parameters
    ----------
    project_dir : path
        ChACRA project root.
    run : int
        Which run number to use (default 1).
    state_idx : int, optional
        Reference state. Defaults to n_states // 2.
    method : {"central", "forward", "backward"}
        Finite-difference rule for dU/dk; see `load_perturbation_energy_femto`.
    save_interval : int, optional
        Contact-frame spacing in cycles for THIS run. Defaults to the value
        in chacra_run.json — which reflects the LATEST run, so it must be
        passed explicitly when scoring an older run recorded with a
        different interval (e.g. run_4 at 10 after run_5 rewrote it to 2).
    """
    from chacra.run_config import RunConfig

    project_dir = Path(project_dir)
    cfg_path = project_dir / "chacra_run.json"
    cfg = RunConfig(str(cfg_path) if cfg_path.exists() else None)

    samples = project_dir / "replica_trajectories" / f"run_{run}" / "samples.arrow"
    if not samples.exists():
        raise FileNotFoundError(f"missing femto samples: {samples}")

    if save_interval is None:
        save_interval = cfg.get("save_interval")
        if save_interval is None:
            raise ValueError(
                f"{cfg_path} has no 'save_interval'; pass save_interval= explicitly."
            )
        save_interval = int(save_interval)
    temps = cfg.get("temps")
    if temps is None:
        raise ValueError(
            f"chacra_run.json at {cfg_path} has no 'temps' — re-run "
            "process-output to populate it."
        )
    temps = np.asarray(temps, dtype=np.float64)
    n_states = len(temps)
    if state_idx is None:
        state_idx = n_states // 2

    # beta of the *Hamiltonian* at the reference state. femto's u_kn are
    # already reduced (multiplied by the relevant beta), so beta = 1.0 in
    # those reduced units. We expose the choice so users with raw energies
    # can override.
    beta = 1.0

    parquet = (
        project_dir / "contact_output" / f"run_{run}" / "contacts"
        / f"cont_state_{state_idx}.parquet"
    )
    if not parquet.exists():
        raise FileNotFoundError(f"missing per-frame contacts: {parquet}")

    phi, contact_names = load_per_frame_contacts(parquet)
    dU, _ = load_perturbation_energy_femto(
        samples, state_idx=state_idx, save_interval=save_interval, method=method,
    )

    # Truncate to common length — femto can have a partial last cycle.
    # A LARGE mismatch is not a partial tail: it means save_interval does not
    # match how this run's contact frames were actually saved, and silently
    # truncating would pair phi frames with the wrong cycles' dU (garbage
    # covariance that still yields a plausible-looking R1).
    n = min(phi.shape[0], dU.shape[0])
    if n < 0.95 * max(phi.shape[0], dU.shape[0]):
        raise ValueError(
            f"phi/dU length mismatch: {phi.shape[0]} contact frames vs "
            f"{dU.shape[0]} strided dU values (save_interval={save_interval}). "
            "chacra_run.json holds the LATEST run's save_interval — pass "
            "save_interval= explicitly for the interval this run was saved at."
        )
    if n != phi.shape[0] or n != dU.shape[0]:
        phi = phi[:n]
        dU = dU[:n]

    return estimate_response(
        phi, dU, beta=beta,
        contact_names=contact_names,
        state_idx=state_idx,
    )
