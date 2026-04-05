"""
Convergence diagnostics for ChACRA HREMD analysis.

Provides metrics to assess whether HREMD contact frequency data has
converged, along with plotting utilities for visual inspection.

All sampling-based metrics (split-half RMSIP, bootstrap) operate on
the **per-frame contact records** — not the pre-computed frequency
matrices.  For each thermodynamic state the raw contacts are read
from ``contact_output/run_*/contacts/cont_state_*.{parquet,tsv}``,
frames are split or resampled, and contact frequencies are
recomputed from each subset.  This properly tests whether the
simulation has sampled enough frames for PCA to be stable.

Metrics
-------
- **RMSIP** (Root Mean Square Inner Product): measures overlap between PCA
  subspaces.  Split-half (within one dataset) or cross-run (comparing
  cumulative frequency matrices between consecutive runs).
- **Contact matrix correlation**: Pearson *r* between mean contact
  probability vectors of consecutive runs.
- **Bootstrap loading stability**: resamples frames to quantify how
  consistently each contact ranks in the top-*k* loadings.
- **Exchange diagnostics**: detects bottlenecks in the replica-exchange
  swap probability matrix.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ───────────────────────────────────────────────────────────────────────────── #
# Per-frame contact data loading                                               #
# ───────────────────────────────────────────────────────────────────────────── #

@dataclass
class _StateContacts:
    """
    Pre-loaded per-frame contact data for one thermodynamic state.

    Stores which frames each residue pair appears in.  This is compact
    (thousands of pairs × hundreds of frames) and allows fast frequency
    calculation for any frame subset.

    Attributes
    ----------
    state_idx : int
        The thermodynamic state index.
    frames : np.ndarray
        Sorted array of unique (globally offset) frame indices.
    pair_frames : dict[str, set[int]]
        Mapping from "res1-res2" → set of frame indices where the pair appears.
    """
    state_idx: int
    frames: np.ndarray
    pair_frames: dict[str, set[int]] = field(default_factory=dict)

    def freq_for_subset(self, frame_set: set[int]) -> dict[str, float]:
        """Compute contact frequencies for a subset of frames."""
        n = len(frame_set)
        if n == 0:
            return {}
        return {
            pair: len(fset & frame_set) / n
            for pair, fset in self.pair_frames.items()
            if len(fset & frame_set) > 0
        }


def _find_contact_files(
    state_idx: int,
    contact_base: str = "./contact_output",
) -> list[tuple[str, str]]:
    """
    Find per-frame contact files for a given state across all runs.

    Returns list of (path, format) where format is ``'parquet'`` or ``'tsv'``.
    """
    base = Path(contact_base)
    files = []
    for run_dir in sorted(base.glob("run_*/contacts")):
        parquet = run_dir / f"cont_state_{state_idx}.parquet"
        tsv = run_dir / f"cont_state_{state_idx}.tsv"
        if parquet.exists():
            files.append((str(parquet), "parquet"))
        elif tsv.exists():
            files.append((str(tsv), "tsv"))
    return files


def _load_state_contacts_from_file(
    path: str, fmt: str, frame_offset: int = 0,
) -> tuple[np.ndarray, dict[str, set[int]]]:
    """
    Read one per-frame contact file and return (frames, pair_frames).

    Parameters
    ----------
    path : str
        Path to contact file (.parquet or .tsv).
    fmt : str
        ``'parquet'`` or ``'tsv'``.
    frame_offset : int
        Offset added to frame indices (for combining files across runs).

    Returns
    -------
    frames : np.ndarray
        Sorted unique (offset) frame indices.
    pair_frames : dict[str, set[int]]
        Mapping from "res1-res2" → set of (offset) frame indices.
    """
    if fmt == "parquet":
        return _load_parquet_contacts(path, frame_offset)
    else:
        return _load_tsv_contacts(path, frame_offset)


def _load_parquet_contacts(
    path: str, frame_offset: int,
) -> tuple[np.ndarray, dict[str, set[int]]]:
    """Stream a parquet contact file via Polars and return pair-frame sets."""
    import polars as pl

    df = (
        pl.scan_parquet(path)
        .with_columns([
            pl.col("atom1").str.split(":").list.slice(0, 3).list.join(":").alias("res1_raw"),
            pl.col("atom2").str.split(":").list.slice(0, 3).list.join(":").alias("res2_raw"),
        ])
        .with_columns([
            pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
              .then(pl.col("res2_raw")).otherwise(pl.col("res1_raw")).alias("res1"),
            pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
              .then(pl.col("res1_raw")).otherwise(pl.col("res2_raw")).alias("res2"),
        ])
        # Deduplicate: one contact per (frame, res1, res2)
        .unique(subset=["frame", "res1", "res2"])
        .select(["frame", "res1", "res2"])
        .collect(streaming=True)
    )

    frames_arr = df["frame"].to_numpy() + frame_offset
    res1_arr = df["res1"].to_list()
    res2_arr = df["res2"].to_list()

    unique_frames = np.unique(frames_arr)
    pair_frames: dict[str, set[int]] = defaultdict(set)
    for frame, r1, r2 in zip(frames_arr, res1_arr, res2_arr):
        pair_frames[f"{r1}-{r2}"].add(int(frame))

    return unique_frames, dict(pair_frames)


def _load_tsv_contacts(
    path: str, frame_offset: int,
) -> tuple[np.ndarray, dict[str, set[int]]]:
    """Stream a getcontacts TSV file and return pair-frame sets."""
    pair_frames: dict[str, set[int]] = defaultdict(set)
    seen_frames: set[int] = set()

    # Track (frame, pair) to deduplicate atom-level contacts to residue-level
    seen_frame_pairs: set[tuple[int, str]] = set()

    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue

            frame = int(parts[0]) + frame_offset
            seen_frames.add(frame)

            # Extract residue labels: chain:resname:resid
            atom1_parts = parts[2].split(":")
            atom2_parts = parts[3].split(":")
            res1 = ":".join(atom1_parts[:3])
            res2 = ":".join(atom2_parts[:3])

            # Canonical ordering
            if res2 < res1:
                res1, res2 = res2, res1

            pair = f"{res1}-{res2}"
            key = (frame, pair)
            if key not in seen_frame_pairs:
                seen_frame_pairs.add(key)
                pair_frames[pair].add(frame)

    return np.array(sorted(seen_frames)), dict(pair_frames)


def _load_state_contacts(
    state_idx: int,
    contact_base: str = "./contact_output",
) -> _StateContacts:
    """
    Load all per-frame contacts for a thermodynamic state, combining
    data from all runs with frame offsets to avoid index collisions.
    """
    files = _find_contact_files(state_idx, contact_base)
    if not files:
        raise FileNotFoundError(
            f"No per-frame contact files found for state {state_idx} "
            f"under {contact_base}/run_*/contacts/"
        )

    all_frames = []
    all_pair_frames: dict[str, set[int]] = defaultdict(set)
    offset = 0

    for path, fmt in files:
        frames, pair_frames = _load_state_contacts_from_file(path, fmt, offset)
        all_frames.append(frames)

        for pair, fset in pair_frames.items():
            all_pair_frames[pair].update(fset)

        # Next run's frames start after this run's max frame + 1
        if len(frames) > 0:
            offset = int(frames.max()) + 1

    return _StateContacts(
        state_idx=state_idx,
        frames=np.concatenate(all_frames) if all_frames else np.array([], dtype=int),
        pair_frames=dict(all_pair_frames),
    )


def _load_state_worker(args: tuple) -> _StateContacts:
    """Multiprocessing wrapper for _load_state_contacts."""
    state_idx, contact_base = args
    return _load_state_contacts(state_idx, contact_base)


def _build_freq_matrix(
    state_data: list[_StateContacts],
    frame_subsets: list[set[int]],
) -> pd.DataFrame:
    """
    Build a contact frequency matrix from per-state frame subsets.

    Parameters
    ----------
    state_data : list[_StateContacts]
        Pre-loaded contact data, one per state (sorted by state_idx).
    frame_subsets : list[set[int]]
        One frame subset per state (same order as state_data).

    Returns
    -------
    pd.DataFrame
        Rows = states, columns = contact pairs, values = frequencies.
    """
    rows = []
    for sc, subset in zip(state_data, frame_subsets):
        rows.append(sc.freq_for_subset(subset))

    df = pd.DataFrame(rows).fillna(0.0)
    df.index = list(range(len(rows)))
    return df


# ───────────────────────────────────────────────────────────────────────────── #
# RMSIP                                                                        #
# ───────────────────────────────────────────────────────────────────────────── #

def rmsip(
    loadings_a: np.ndarray,
    loadings_b: np.ndarray,
    k: int,
) -> float:
    """
    Root Mean Square Inner Product between two PCA subspaces.

    Parameters
    ----------
    loadings_a, loadings_b : np.ndarray
        Component matrices of shape (n_contacts, n_components).
        Columns are eigenvectors (PC loading vectors).
    k : int
        Number of leading PCs to compare.

    Returns
    -------
    float
        RMSIP value in [0, 1].  1.0 = identical subspaces.
    """
    U = loadings_a[:, :k].copy()
    V = loadings_b[:, :k].copy()

    # Ensure columns are unit vectors
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)

    overlap = (U.T @ V) ** 2  # shape (k, k)
    return float(min(1.0, np.sqrt(overlap.sum() / k)))


def _fit_pca_loadings(contact_df: pd.DataFrame) -> np.ndarray:
    """Fit PCA and return component matrix (n_contacts, n_components)."""
    pca = PCA()
    pca.fit(contact_df)
    return pca.components_.T  # (n_contacts, n_components)


def split_half_rmsip(
    n_states: int,
    k: int = 3,
    contact_base: str = "./contact_output",
    n_jobs: int = 4,
) -> float:
    """
    Split-half RMSIP computed from per-frame contact records.

    For each thermodynamic state, the per-frame contacts from all runs
    are loaded and the frames are split into two chronological halves.
    Contact frequencies are independently recomputed for each half,
    producing two temperature-dependent frequency matrices.  PCA is
    fitted on each and the RMSIP measures subspace overlap.

    This tests whether the simulation has accumulated enough frames
    for the PCA decomposition to be stable.

    Parameters
    ----------
    n_states : int
        Number of thermodynamic states (replicas).
    k : int
        Number of leading PCs to compare.
    contact_base : str
        Root path to contact output (contains ``run_*/contacts/``).
    n_jobs : int
        Number of parallel workers for loading contact data.

    Returns
    -------
    float
        RMSIP value in [0, 1].
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = cpu_count()

    # Phase 1: Load per-frame data for all states (parallel)
    args = [(i, contact_base) for i in range(n_states)]
    with Pool(min(n_jobs, n_states)) as pool:
        state_data = pool.map(_load_state_worker, args)

    # Sort by state index
    state_data.sort(key=lambda sc: sc.state_idx)

    # Phase 2: Split frames and compute frequencies
    subsets_a = []
    subsets_b = []
    for sc in state_data:
        n = len(sc.frames)
        if n < 4:
            return float("nan")
        mid = n // 2
        subsets_a.append(set(sc.frames[:mid].tolist()))
        subsets_b.append(set(sc.frames[mid:].tolist()))

    df_a = _build_freq_matrix(state_data, subsets_a)
    df_b = _build_freq_matrix(state_data, subsets_b)

    # Align columns
    all_cols = df_a.columns.union(df_b.columns)
    df_a = df_a.reindex(columns=all_cols, fill_value=0.0)
    df_b = df_b.reindex(columns=all_cols, fill_value=0.0)

    # Phase 3: PCA + RMSIP
    loadings_a = _fit_pca_loadings(df_a)
    loadings_b = _fit_pca_loadings(df_b)

    k = min(k, df_a.shape[0], df_b.shape[0], len(all_cols))
    return rmsip(loadings_a, loadings_b, k)


def cross_run_rmsip(
    run: int,
    k: int = 3,
    analysis_dir: str = "./analysis_output",
) -> float | None:
    """
    Compare PCA subspaces between cumulative data at run N-1 and run N.

    Uses the pre-computed ``total_contacts.parquet`` from each run's
    analysis output (these are the weighted cumulative frequency matrices).

    Parameters
    ----------
    run : int
        Current run number (must be >= 2).
    k : int
        Number of leading PCs to compare.
    analysis_dir : str
        Path to the analysis output root.

    Returns
    -------
    float or None
        RMSIP value, or None if run-1 data doesn't exist.
    """
    if run < 2:
        return None

    prior = Path(analysis_dir) / f"run_{run - 1}" / "total_contacts.parquet"
    current = Path(analysis_dir) / f"run_{run}" / "total_contacts.parquet"

    if not prior.exists() or not current.exists():
        return None

    df_prior = pd.read_parquet(prior)
    df_current = pd.read_parquet(current)

    all_cols = df_prior.columns.union(df_current.columns)
    df_prior = df_prior.reindex(columns=all_cols, fill_value=0.0)
    df_current = df_current.reindex(columns=all_cols, fill_value=0.0)

    loadings_a = _fit_pca_loadings(df_prior)
    loadings_b = _fit_pca_loadings(df_current)

    k = min(k, df_prior.shape[0], df_current.shape[0], len(all_cols))
    return rmsip(loadings_a, loadings_b, k)


# ───────────────────────────────────────────────────────────────────────────── #
# Contact matrix correlation                                                   #
# ───────────────────────────────────────────────────────────────────────────── #

def contact_matrix_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> float:
    """
    Pearson correlation between the mean contact probability vectors
    of two frequency matrices.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Contact frequency matrices (rows = states, columns = contacts).

    Returns
    -------
    float
        Pearson *r* in [-1, 1].  Values > 0.99 indicate a stable input matrix.
    """
    all_cols = df_a.columns.union(df_b.columns)
    mean_a = df_a.reindex(columns=all_cols, fill_value=0.0).mean(axis=0)
    mean_b = df_b.reindex(columns=all_cols, fill_value=0.0).mean(axis=0)

    if mean_a.std() == 0 or mean_b.std() == 0:
        return float("nan")

    return float(np.corrcoef(mean_a.values, mean_b.values)[0, 1])


def cross_run_contact_correlation(
    run: int,
    analysis_dir: str = "./analysis_output",
) -> float | None:
    """
    Pearson correlation of mean contact vectors between run N-1 and run N.
    Uses pre-computed total_contacts.parquet from each run.
    """
    if run < 2:
        return None

    prior = Path(analysis_dir) / f"run_{run - 1}" / "total_contacts.parquet"
    current = Path(analysis_dir) / f"run_{run}" / "total_contacts.parquet"

    if not prior.exists() or not current.exists():
        return None

    return contact_matrix_correlation(
        pd.read_parquet(prior),
        pd.read_parquet(current),
    )


# ───────────────────────────────────────────────────────────────────────────── #
# Bootstrap loading stability                                                  #
# ───────────────────────────────────────────────────────────────────────────── #

def _bootstrap_iteration(args: tuple) -> dict[str, list[str]]:
    """
    Worker for one bootstrap iteration.

    Parameters
    ----------
    args : tuple
        (state_data, k, n_top, seed)

    Returns
    -------
    dict mapping "PC{i}" → list of top-n_top contact names.
    """
    state_data, k, n_top, seed = args
    rng = np.random.default_rng(seed)

    subsets = []
    for sc in state_data:
        n = len(sc.frames)
        resampled = rng.choice(sc.frames, size=n, replace=True)
        subsets.append(set(resampled.tolist()))

    df = _build_freq_matrix(state_data, subsets)
    if df.shape[1] < k:
        return {}

    pca = PCA(n_components=min(k, df.shape[0], df.shape[1]))
    pca.fit(df)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    result = {}
    for pc in range(1, pca.n_components_ + 1):
        col = f"PC{pc}"
        top = loadings[col].abs().nlargest(n_top).index.tolist()
        result[col] = top

    return result


def bootstrap_loadings(
    n_states: int,
    k: int = 3,
    n_top: int = 20,
    n_bootstrap: int = 100,
    contact_base: str = "./contact_output",
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Assess loading stability by bootstrap resampling of per-frame contacts.

    For each bootstrap iteration, frames are resampled with replacement
    independently within each thermodynamic state, contact frequencies
    are recomputed, and PCA is fitted.  The metric is **rank stability**:
    how frequently each contact appears in the top-*n_top* loadings for
    each PC across bootstrap samples.

    Parameters
    ----------
    n_states : int
        Number of thermodynamic states.
    k : int
        Number of PCs to evaluate.
    n_top : int
        How many top contacts to track per PC.
    n_bootstrap : int
        Number of bootstrap resamples.
    contact_base : str
        Root path to contact output.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    pd.DataFrame
        Columns: PC1_rank_freq, PC2_rank_freq, ...
        Index: contact names.
        Values: fraction of bootstraps in which the contact appeared
        in the top-*n_top* for that PC (0.0–1.0).
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = cpu_count()

    # Phase 1: Load all contact data (parallel)
    print("  [bootstrap] Loading per-frame contact data...")
    load_args = [(i, contact_base) for i in range(n_states)]
    with Pool(min(n_jobs, n_states)) as pool:
        state_data = pool.map(_load_state_worker, load_args)
    state_data.sort(key=lambda sc: sc.state_idx)

    # Collect all contact names
    all_contacts = set()
    for sc in state_data:
        all_contacts.update(sc.pair_frames.keys())
    contacts = sorted(all_contacts)

    print(f"  [bootstrap] Loaded {len(contacts)} contact pairs across {n_states} states.")
    print(f"  [bootstrap] Running {n_bootstrap} bootstrap iterations...")

    # Phase 2: Bootstrap iterations
    # NOTE: We can't easily pickle _StateContacts for multiprocessing
    # because the pair_frames dicts are large.  Run iterations sequentially
    # but each iteration is fast since data is in memory.
    counts = {
        f"PC{pc}_rank_freq": pd.Series(0.0, index=contacts)
        for pc in range(1, k + 1)
    }

    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        subsets = []
        for sc in state_data:
            n = len(sc.frames)
            resampled = rng.choice(sc.frames, size=n, replace=True)
            subsets.append(set(resampled.tolist()))

        df = _build_freq_matrix(state_data, subsets)
        if df.shape[1] < k:
            continue

        pca = PCA(n_components=min(k, df.shape[0], df.shape[1]))
        pca.fit(df)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=df.columns,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        )

        for pc in range(1, pca.n_components_ + 1):
            col = f"PC{pc}"
            top = loadings[col].abs().nlargest(n_top).index
            counts[f"{col}_rank_freq"][top] += 1

    # Normalize to frequency
    result = pd.DataFrame(counts) / n_bootstrap
    result = result.reindex(contacts).fillna(0.0)
    result["max_freq"] = result.max(axis=1)
    result = result.sort_values("max_freq", ascending=False).drop(columns="max_freq")
    return result


# ───────────────────────────────────────────────────────────────────────────── #
# Exchange diagnostics                                                         #
# ───────────────────────────────────────────────────────────────────────────── #

def exchange_diagnostics(exchange_probs: np.ndarray) -> dict:
    """
    Analyze exchange probability array for bottlenecks.

    Parameters
    ----------
    exchange_probs : np.ndarray
        1-D array of nearest-neighbor exchange probabilities
        (length = n_states - 1).

    Returns
    -------
    dict with keys:
        - ``min_prob``: minimum swap probability
        - ``mean_prob``: mean swap probability
        - ``bottlenecks``: state pairs with prob < 0.10
        - ``estimated_round_trip``: estimated cycles for a full
          λ-ladder traversal
    """
    result = {
        "min_prob": float(np.nanmin(exchange_probs)),
        "mean_prob": float(np.nanmean(exchange_probs)),
        "bottlenecks": [],
        "estimated_round_trip": float("nan"),
    }

    for i, p in enumerate(exchange_probs):
        if p < 0.10:
            result["bottlenecks"].append({
                "states": [i, i + 1],
                "probability": round(float(p), 4),
            })

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_probs = np.where(exchange_probs > 0, exchange_probs, np.nan)
        half_trip = np.nansum(1.0 / safe_probs)
        result["estimated_round_trip"] = round(float(2 * half_trip), 1)

    return result


# ───────────────────────────────────────────────────────────────────────────── #
# Main convergence report                                                      #
# ───────────────────────────────────────────────────────────────────────────── #

_THRESHOLDS = {
    "rmsip_excellent": 0.85,
    "rmsip_good": 0.70,
    "rmsip_developing": 0.50,
    "correlation_stable": 0.99,
    "correlation_warn": 0.98,
}


def _rmsip_label(val: float | None) -> str:
    if val is None or np.isnan(val):
        return "n/a"
    if val >= _THRESHOLDS["rmsip_excellent"]:
        return "converged"
    if val >= _THRESHOLDS["rmsip_good"]:
        return "good"
    if val >= _THRESHOLDS["rmsip_developing"]:
        return "developing"
    return "poor"


def _correlation_label(val: float | None) -> str:
    if val is None or np.isnan(val):
        return "n/a"
    if val >= _THRESHOLDS["correlation_stable"]:
        return "stable"
    if val >= _THRESHOLDS["correlation_warn"]:
        return "marginal"
    return "unstable"


def _determine_verdict(report: dict) -> str:
    """Produce an overall verdict string from the individual metrics."""
    sh = report.get("split_half_rmsip")
    cr = report.get("cross_run_rmsip")
    cc = report.get("contact_correlation")
    bottlenecks = report.get("exchange_bottlenecks", [])

    score = 0

    if sh is not None and not np.isnan(sh):
        if sh >= 0.85:
            score += 3
        elif sh >= 0.70:
            score += 2
        elif sh >= 0.50:
            score += 1

    if cr is not None and not np.isnan(cr):
        if cr >= 0.85:
            score += 3
        elif cr >= 0.70:
            score += 2
        elif cr >= 0.50:
            score += 1

    if cc is not None and not np.isnan(cc):
        if cc >= 0.99:
            score += 2
        elif cc >= 0.98:
            score += 1

    if len(bottlenecks) > 0:
        score -= 1

    has_cross_run = (cr is not None and not np.isnan(cr))
    max_score = 8 if has_cross_run else 3
    ratio = score / max_score if max_score > 0 else 0

    if ratio >= 0.8:
        return "converged"
    elif ratio >= 0.6:
        return "likely_converged"
    elif ratio >= 0.3:
        return "developing"
    else:
        return "not_converged"


def convergence_report(
    run: int,
    n_states: int,
    exchange_probs: np.ndarray | None = None,
    k: int = 3,
    contact_base: str = "./contact_output",
    analysis_dir: str = "./analysis_output",
    n_jobs: int = 4,
) -> dict:
    """
    Compute all convergence metrics and return a structured report.

    The split-half RMSIP is computed from the per-frame contact records
    (not the pre-computed frequency matrix), properly testing whether
    sampling is sufficient for stable PCA.

    Parameters
    ----------
    run : int
        Current run number.
    n_states : int
        Number of thermodynamic states.
    exchange_probs : np.ndarray or None
        Nearest-neighbor swap probabilities.
    k : int
        Number of leading PCs to compare in RMSIP.
    contact_base : str
        Root path to contact output.
    analysis_dir : str
        Path to analysis_output root.
    n_jobs : int
        Number of parallel workers for loading contact data.

    Returns
    -------
    dict
        Full convergence report with metrics and verdict.
    """
    report = {"run": run, "k": k}

    # 1. Split-half RMSIP (from per-frame contacts)
    try:
        sh = split_half_rmsip(
            n_states, k=k, contact_base=contact_base, n_jobs=n_jobs,
        )
        report["split_half_rmsip"] = round(sh, 4) if not np.isnan(sh) else None
    except (FileNotFoundError, ValueError) as e:
        print(f"  [convergence] Split-half RMSIP skipped: {e}")
        report["split_half_rmsip"] = None
    report["split_half_label"] = _rmsip_label(report["split_half_rmsip"])

    # 2. Cross-run RMSIP (from cumulative frequency matrices)
    cr = cross_run_rmsip(run, k=k, analysis_dir=analysis_dir)
    report["cross_run_rmsip"] = round(cr, 4) if cr is not None else None
    report["cross_run_label"] = _rmsip_label(cr)

    # 3. Contact matrix correlation
    cc = cross_run_contact_correlation(run, analysis_dir=analysis_dir)
    report["contact_correlation"] = round(cc, 4) if cc is not None else None
    report["contact_correlation_label"] = _correlation_label(cc)

    # 4. Exchange diagnostics
    if exchange_probs is not None:
        ex = exchange_diagnostics(exchange_probs)
        report["exchange_min_prob"] = ex["min_prob"]
        report["exchange_mean_prob"] = ex["mean_prob"]
        report["exchange_bottlenecks"] = ex["bottlenecks"]
        report["estimated_round_trip_cycles"] = ex["estimated_round_trip"]
    else:
        report["exchange_min_prob"] = None
        report["exchange_mean_prob"] = None
        report["exchange_bottlenecks"] = []
        report["estimated_round_trip_cycles"] = None

    # 5. Verdict
    report["verdict"] = _determine_verdict(report)

    return report


def print_convergence_report(report: dict) -> None:
    """Pretty-print a convergence report to stdout."""
    run = report["run"]
    k = report["k"]

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Convergence Report — Run {run:<4}  (top {k} PCs)     │")
    print(f"  └─────────────────────────────────────────────────┘")

    sh = report.get("split_half_rmsip")
    sh_lbl = report.get("split_half_label", "n/a")
    sh_str = f"{sh:.3f}" if sh is not None else "n/a"
    print(f"    Split-half RMSIP:       {sh_str:>8}   — {sh_lbl}")

    cr = report.get("cross_run_rmsip")
    cr_lbl = report.get("cross_run_label", "n/a")
    if cr is not None:
        print(f"    Cross-run RMSIP:        {cr:.3f}   — {cr_lbl}")
    else:
        print(f"    Cross-run RMSIP:             n/a   (run 1; need run 2+)")

    cc = report.get("contact_correlation")
    cc_lbl = report.get("contact_correlation_label", "n/a")
    if cc is not None:
        print(f"    Contact correlation:     {cc:.4f}  — {cc_lbl}")
    else:
        print(f"    Contact correlation:          n/a   (run 1; need run 2+)")

    ex_min = report.get("exchange_min_prob")
    if ex_min is not None:
        ex_mean = report.get("exchange_mean_prob", 0)
        print(f"    Exchange prob (min/mean): {ex_min:.3f} / {ex_mean:.3f}")
        bottlenecks = report.get("exchange_bottlenecks", [])
        if bottlenecks:
            pairs = ", ".join(
                f"{b['states'][0]}↔{b['states'][1]}({b['probability']:.2f})"
                for b in bottlenecks
            )
            print(f"    Exchange bottlenecks:    {pairs}")
        else:
            print(f"    Exchange bottlenecks:     none")
        rt = report.get("estimated_round_trip_cycles")
        if rt is not None:
            print(f"    Est. round-trip cycles:   ~{rt:.0f}")

    verdict = report.get("verdict", "unknown")
    verdicts_display = {
        "converged": "✓ CONVERGED",
        "likely_converged": "● LIKELY CONVERGED — consider one more run to confirm",
        "developing": "○ DEVELOPING — more sampling needed",
        "not_converged": "✗ NOT CONVERGED — continue running",
    }
    print(f"\n    Verdict: {verdicts_display.get(verdict, verdict)}")
    print()


def save_convergence_report(
    report: dict,
    output_dir: str,
) -> str:
    """Write report to JSON.  Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "convergence.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


# ───────────────────────────────────────────────────────────────────────────── #
# Plotting                                                                     #
# ───────────────────────────────────────────────────────────────────────────── #

def plot_convergence_history(
    analysis_dir: str = "./analysis_output",
    filename: str | os.PathLike | None = None,
) -> plt.Figure | None:
    """
    Plot RMSIP and contact correlation across all runs that have
    a ``convergence.json``.
    """
    runs, sh_vals, cr_vals, cc_vals = [], [], [], []

    analysis_path = Path(analysis_dir)
    if not analysis_path.exists():
        print("[convergence] Analysis directory not found.")
        return None
    for d in sorted(analysis_path.iterdir()):
        if not d.is_dir():
            continue
        conv = d / "convergence.json"
        if conv.exists():
            with open(conv) as f:
                data = json.load(f)
            runs.append(data["run"])
            sh_vals.append(data.get("split_half_rmsip"))
            cr_vals.append(data.get("cross_run_rmsip"))
            cc_vals.append(data.get("contact_correlation"))

    if len(runs) < 1:
        print("[convergence] No convergence.json files found.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(runs, sh_vals, "o-", label="Split-half RMSIP", color="#2196F3")
    if any(v is not None for v in cr_vals):
        ax.plot(runs, cr_vals, "s-", label="Cross-run RMSIP", color="#FF9800")
    ax.axhline(0.85, color="green", linestyle="--", alpha=0.5, label="Excellent (0.85)")
    ax.axhline(0.70, color="orange", linestyle="--", alpha=0.5, label="Good (0.70)")
    ax.set_xlabel("Run")
    ax.set_ylabel("RMSIP")
    ax.set_title("PCA Subspace Convergence")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_xticks(runs)

    ax = axes[1]
    if any(v is not None for v in cc_vals):
        ax.plot(runs, cc_vals, "D-", color="#4CAF50", label="Contact correlation")
        ax.axhline(0.99, color="green", linestyle="--", alpha=0.5, label="Stable (0.99)")
        ax.axhline(0.98, color="orange", linestyle="--", alpha=0.5, label="Marginal (0.98)")
        ax.set_ylim(0.90, 1.005)
    else:
        ax.text(
            0.5, 0.5, "Available from run 2+",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
            color="gray",
        )
    ax.set_xlabel("Run")
    ax.set_ylabel("Pearson r")
    ax.set_title("Contact Matrix Stability")
    ax.legend(fontsize=8)
    ax.set_xticks(runs)

    if filename is not None:
        fig.savefig(filename, dpi=150)

    return fig


def plot_exchange_diagnostics(
    exchange_probs: np.ndarray,
    filename: str | os.PathLike | None = None,
) -> plt.Figure:
    """
    Bar chart of nearest-neighbor exchange probabilities with
    bottleneck highlighting.
    """
    n = len(exchange_probs)
    labels = [f"{i}↔{i+1}" for i in range(n)]
    colors = [
        "#f44336" if p < 0.10 else ("#FF9800" if p < 0.20 else "#4CAF50")
        for p in exchange_probs
    ]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), 4), constrained_layout=True)
    ax.bar(labels, exchange_probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.10, color="red", linestyle="--", alpha=0.6, label="Bottleneck (< 0.10)")
    ax.axhline(0.20, color="orange", linestyle="--", alpha=0.4, label="Low (< 0.20)")
    ax.set_xlabel("State pair")
    ax.set_ylabel("Swap probability")
    ax.set_title("Replica Exchange Swap Probabilities")
    ax.set_ylim(0, min(1.0, exchange_probs.max() * 1.3) if n > 0 else 1.0)
    ax.legend(fontsize=8)

    if n > 12:
        ax.tick_params(axis="x", rotation=45)

    if filename is not None:
        fig.savefig(filename, dpi=150)

    return fig
