"""
MBAR-pooled response estimators over the full HREMD ladder.

Why
---
The single-state cumulant estimator in :mod:`chacra.response` uses the frames
of ONE replica.  femto evaluates every sampled configuration under every
state's Hamiltonian (the ``u_kn`` matrix in ``samples.arrow``), which is exactly
the input MBAR needs to reweight *all* frames of *all* states to any target
Hamiltonian.  On the AdK benchmark this takes the split-half self-consistency
of R^(1) from ~0.6 to ~0.997 with no additional simulation.

REST2 structure
---------------
femto's REST2 scaling makes the reduced potential of a fixed configuration an
exact function of the scaling parameter lambda = beta_m / beta_0::

    u(x; lambda) = a(x) + b(x) * lambda + c(x) * sqrt(lambda)

(a: solvent-solvent + pV, b: solute-solute torsions/electrostatics/LJ,
c: solute-solvent).  Fitting a, b, c per frame from the K stored evaluations
gives u at *any* lambda, and the generators::

    du/dlambda   = b + c / (2 sqrt(lambda))
    d2u/dlambda2 = -c / (4 lambda^(3/2))

Response functions (reduced units, beta absorbed into u)::

    <phi>(lambda)      = sum_n w_n(lambda) phi_n
    R1 = d<phi>/dlambda   = -Cov_w(phi, du/dlambda)
    R2 = d2<phi>/dlambda2 = <dphi (d du/dlambda)^2>_w,c - Cov_w(phi, d2u/dlambda2)

Layout
------
- :class:`OccupancyStore` — per-state per-frame contact occupancy (uint8) plus
  per-frame reduced potentials under every state; cached as one ``.npz``.
- :func:`build_occupancy_store` — from a ChACRA run directory (per-state
  ``cont_state_*.parquet`` or per-replica ``cont_replica_*.parquet``).
- :func:`mbar_free_energies`, :func:`mbar_log_weights` — self-consistent MBAR
  (pymbar is used for refinement when importable).
- :func:`response_at_lambda`, :func:`response_curve` — R1/R2 at arbitrary lambda.
- :func:`bootstrap_response` — block-bootstrap uncertainties.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp


# --------------------------------------------------------------------------- #
# Occupancy store                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class OccupancyStore:
    """
    Per-frame data for one HREMD run.

    Attributes
    ----------
    phi : (K, n_frames, C) uint8
        Contact occupancy of the configuration sampled at state ``s`` on
        saved frame ``j``.
    u : (K, n_frames, K) float64
        Reduced potential of that configuration under every state's
        Hamiltonian (last axis).  ``u[s, j, s]`` is the energy under the
        state it was sampled at.
    contacts : list[str]
        Contact names ("chain:resn:resid-chain:resn:resid"), length C.
    lambdas : (K,) float
        beta_m / beta_0 of each state (1.0 for the unscaled state).
    temps : (K,) float
        Effective temperatures (K).
    """

    phi: np.ndarray
    u: np.ndarray
    contacts: list[str]
    lambdas: np.ndarray
    temps: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.phi.shape[0]

    @property
    def n_frames(self) -> int:
        return self.phi.shape[1]

    @property
    def n_contacts(self) -> int:
        return self.phi.shape[2]

    # -- persistence -------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        import json

        np.savez_compressed(
            str(path),
            phi=self.phi,
            u=self.u,
            contacts=np.array(self.contacts, dtype=object),
            lambdas=self.lambdas,
            temps=self.temps,
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "OccupancyStore":
        import json

        with np.load(str(path), allow_pickle=True) as z:
            return cls(
                phi=z["phi"],
                u=z["u"],
                contacts=[str(c) for c in z["contacts"]],
                lambdas=z["lambdas"],
                temps=z["temps"],
                meta=json.loads(str(z["meta"])),
            )

    # -- derived quantities --------------------------------------------------
    def freqs(self) -> pd.DataFrame:
        """Per-state contact frequencies (rows = states)."""
        return pd.DataFrame(
            self.phi.mean(axis=1, dtype=np.float64),
            index=np.round(self.temps).astype(int),
            columns=self.contacts,
        )

    def pooled_u_kn(self) -> np.ndarray:
        """(K, N) reduced potentials in MBAR layout, N = K * n_frames."""
        return np.concatenate([self.u[s] for s in range(self.n_states)], axis=0).T

    def energy_decomposition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Fit u(x; lambda) = a + b*lambda + c*sqrt(lambda) per frame.

        Returns (a, b, c) each of shape (N,) in pooled order, and the maximum
        absolute residual of the fit (should be ~1e-6 for femto REST2 output;
        a large value means the Hamiltonian is not of REST2 form).
        """
        lam = np.asarray(self.lambdas, dtype=np.float64)
        A = np.stack([np.ones_like(lam), lam, np.sqrt(lam)], axis=1)  # (K, 3)
        U = self.pooled_u_kn()                                          # (K, N)
        coef, *_ = np.linalg.lstsq(A, U, rcond=None)                    # (3, N)
        resid = float(np.abs(A @ coef - U).max())
        return coef[0], coef[1], coef[2], resid


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #


def _residue_pairs_from_long_parquet(path: str | os.PathLike):
    """
    Read an ultracontacts long-format parquet with dictionary-encoded atom
    columns and return (frame_idx, pair_name) arrays with atoms collapsed to
    residues and pairs canonically ordered.  Memory ~ O(rows) ints, not
    O(rows) strings.
    """
    import pyarrow.parquet as pq

    t = pq.read_table(
        str(path), columns=["frame", "atom1", "atom2"],
        read_dictionary=["atom1", "atom2"],
    )
    frames = t.column("frame").to_numpy()

    def residue_codes(col):
        col = col.combine_chunks()
        dic = np.array(
            [":".join(x.split(":")[:3]) for x in col.dictionary.to_pylist()],
            dtype=object,
        )
        return dic[col.indices.to_numpy()]

    a = residue_codes(t.column("atom1"))
    b = residue_codes(t.column("atom2"))
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    pairs = pd.Series(lo).str.cat(pd.Series(hi), sep="-").to_numpy()
    # Compress to (unique pair names, int32 codes) so a 5M-event file costs
    # ~20 MB instead of hundreds of MB of Python strings.
    uniq, inv = np.unique(pairs, return_inverse=True)
    return frames.astype(np.int64), uniq, inv.astype(np.int32)


def _residue_pairs_from_tsv(path: str | os.PathLike):
    """getcontacts per-frame TSV → (frames, unique pair names, int32 codes)."""
    frames, pairs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                fr = int(parts[0])
            except ValueError:
                continue
            r1 = ":".join(parts[2].split(":")[:3])
            r2 = ":".join(parts[3].split(":")[:3])
            frames.append(fr)
            pairs.append(f"{r1}-{r2}" if r1 <= r2 else f"{r2}-{r1}")
    frames = np.asarray(frames, dtype=np.int64)
    uniq, inv = np.unique(np.asarray(pairs, dtype=object), return_inverse=True)
    return frames, uniq, inv.astype(np.int32)


def _residue_pairs(path: str | os.PathLike):
    path = str(path)
    if path.endswith(".parquet"):
        return _residue_pairs_from_long_parquet(path)
    return _residue_pairs_from_tsv(path)


def _occupancy_from_pairs(frames, uniq, inv, contact_index: dict, n_frames: int) -> np.ndarray:
    col_of_uniq = np.array([contact_index.get(p, -1) for p in uniq])
    cols = col_of_uniq[inv]
    ok = (cols >= 0) & (frames < n_frames)
    phi = np.zeros((n_frames, len(contact_index)), dtype=np.uint8)
    phi[frames[ok], cols[ok]] = 1
    return phi


def load_femto_reduced_potentials(
    samples_arrow: str | os.PathLike, save_interval: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (u_state, r2s) for the saved frames of a run.

    u_state : (K, n_frames, K)
        u_state[s, j, k] = reduced potential under Hamiltonian k of the
        configuration that was at state s on saved frame j.
    r2s : (n_frames, K) int
        replica -> state map on each saved frame.
    """
    import pyarrow

    from chacra.trajectories.process_hremd import saved_frame_rows

    with pyarrow.OSFile(str(samples_arrow), "rb") as fh, \
         pyarrow.RecordBatchStreamReader(fh) as reader:
        tbl = reader.read_all()
    n_rows = tbl.num_rows
    K = len(tbl.column("replica_to_state_idx")[0])
    u_all = (
        tbl.column("u_kn").combine_chunks().flatten().flatten()
        .to_numpy().reshape(n_rows, K, K)
    )  # [row, hamiltonian k, replica m]
    r2s_all = (
        tbl.column("replica_to_state_idx").combine_chunks().flatten()
        .to_numpy().reshape(n_rows, K).astype(int)
    )
    rows = saved_frame_rows(pd.DataFrame({"step": tbl.column("step").to_numpy()}),
                            save_interval)
    u_all = u_all[rows]
    r2s = r2s_all[rows]
    n_fr = len(rows)
    u_state = np.empty((K, n_fr, K), dtype=np.float64)
    for s in range(K):
        rep = np.argmax(r2s == s, axis=1)
        u_state[s] = u_all[np.arange(n_fr), :, rep]
    return u_state, r2s


def build_occupancy_store(
    project_dir: str | os.PathLike,
    run: int,
    save_interval: int | None = None,
    contacts_dir: str | os.PathLike | None = None,
    n_frames: int | None = None,
    verbose: bool = True,
) -> OccupancyStore:
    """
    Build an :class:`OccupancyStore` for ``run`` of a ChACRA project.

    Per-frame contacts are read from ``contact_output/run_N/contacts/`` —
    either per-state files ``cont_state_{s}.parquet`` (legacy demux path) or
    per-replica files ``cont_replica_{r}.parquet`` (contacts computed
    directly on replica trajectories; frames are assigned to states with the
    exchange record).

    ``n_frames`` (the replica trajectory length) is authoritative when given;
    otherwise the frame count is inferred from the contact files and must
    agree with the number of saved energy rows to within 5%.
    """
    from chacra.run_config import RunConfig

    project_dir = Path(project_dir)
    cfg_path = project_dir / "chacra_run.json"
    cfg = RunConfig(str(cfg_path) if cfg_path.exists() else None)
    if save_interval is None:
        save_interval = int(cfg.get("save_interval") or 1)
    temps = cfg.get("temps")
    if temps is None:
        raise ValueError(f"{cfg_path} has no 'temps'; run process-output first.")
    temps = np.asarray(temps, dtype=np.float64)
    lambdas = temps[0] / temps
    K = len(temps)

    samples = project_dir / "replica_trajectories" / f"run_{run}" / "samples.arrow"
    if not samples.exists():
        raise FileNotFoundError(samples)
    u_state, r2s = load_femto_reduced_potentials(samples, save_interval)
    n_fr_u = u_state.shape[1]
    if verbose:
        print(f"  [reweight] {n_fr_u} saved frames x {K} states from {samples.name}")

    cdir = Path(contacts_dir) if contacts_dir else (
        project_dir / "contact_output" / f"run_{run}" / "contacts"
    )
    def _find(prefix):
        out = []
        for i in range(K):
            for ext in (".parquet", ".tsv"):
                p = cdir / f"{prefix}_{i}{ext}"
                if p.exists():
                    out.append(p)
                    break
        return out if len(out) == K else None

    replica_files = _find("cont_replica")
    state_files = _find("cont_state")
    if replica_files:
        mode, files = "replica", replica_files
    elif state_files:
        mode, files = "state", state_files
    else:
        raise FileNotFoundError(
            f"need all cont_replica_*.{{parquet,tsv}} or all cont_state_*.{{parquet,tsv}} in {cdir}"
        )

    # Pass 1: read pairs, collect the contact universe and frame counts.
    raw = []
    contact_set: set[str] = set()
    n_fr_c = 0
    for p in files:
        frames, uniq, inv = _residue_pairs(p)
        raw.append((frames, uniq, inv))
        contact_set.update(uniq.tolist())
        n_fr_c = max(n_fr_c, int(frames.max()) + 1 if len(frames) else 0)
        if verbose:
            print(f"  [reweight] {p.name}: {len(frames)} events, {len(contact_set)} contacts so far")
    contacts = sorted(contact_set)
    cidx = {c: i for i, c in enumerate(contacts)}

    if n_frames is not None:
        n_fr = int(n_frames)
        if n_fr_u < n_fr:
            raise ValueError(
                f"trajectory has {n_fr} frames but only {n_fr_u} saved energy rows "
                f"(save_interval={save_interval}); wrong save_interval?"
            )
    else:
        n_fr = min(n_fr_u, n_fr_c)
        if n_fr < 0.95 * max(n_fr_u, n_fr_c):
            raise ValueError(
                f"frame count mismatch: {n_fr_c} contact frames vs {n_fr_u} saved "
                f"energy rows (save_interval={save_interval}). Pass the save_interval "
                "this run was actually written with, or n_frames."
            )
    u_state = u_state[:, :n_fr]
    r2s = r2s[:n_fr]

    phi = np.zeros((K, n_fr, len(contacts)), dtype=np.uint8)
    if mode == "state":
        for s, (frames, uniq, inv) in enumerate(raw):
            phi[s] = _occupancy_from_pairs(frames, uniq, inv, cidx, n_fr)
    else:
        # replica r's frame j belongs to state r2s[j, r]
        for r, (frames, uniq, inv) in enumerate(raw):
            phi_r = _occupancy_from_pairs(frames, uniq, inv, cidx, n_fr)
            states_of_frames = r2s[:, r]
            for s in range(K):
                sel = states_of_frames == s
                phi[s, sel] = phi_r[sel]
    del raw

    return OccupancyStore(
        phi=phi, u=u_state, contacts=contacts, lambdas=lambdas, temps=temps,
        meta={"run": int(run), "save_interval": int(save_interval),
              "contact_source": mode, "n_frames": int(n_fr)},
    )


def default_store_path(project_dir: str | os.PathLike, run: int) -> Path:
    return Path(project_dir) / "contact_output" / f"run_{run}" / "occupancy.npz"


def get_occupancy_store(
    project_dir: str | os.PathLike, run: int, rebuild: bool = False, **kw
) -> OccupancyStore:
    """Load the cached store for a run, building (and caching) it if needed."""
    path = default_store_path(project_dir, run)
    if path.exists() and not rebuild:
        return OccupancyStore.load(path)
    store = build_occupancy_store(project_dir, run, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save(path)
    return store


# --------------------------------------------------------------------------- #
# MBAR                                                                         #
# --------------------------------------------------------------------------- #


def _initial_free_energies(u_kn: np.ndarray, N_k: np.ndarray) -> np.ndarray:
    """
    Sequential exponential-averaging estimate of f_k, used to seed the
    self-consistent iteration.  Zero initial guesses do not converge when
    adjacent-state gaps are hundreds of kT (typical for a whole-protein
    REST2 ladder).
    """
    K = u_kn.shape[0]
    starts = np.concatenate([[0], np.cumsum(N_k)])
    f = np.zeros(K)
    for k in range(K - 1):
        sl = slice(starts[k], starts[k + 1])
        d = u_kn[k + 1, sl] - u_kn[k, sl]
        f[k + 1] = f[k] - (logsumexp(-d) - np.log(N_k[k]))
    return f


def mbar_free_energies(
    u_kn: np.ndarray,
    N_k: np.ndarray,
    f_init: np.ndarray | None = None,
    tol: float = 1e-7,
    max_iter: int = 3000,
    use_pymbar: bool = True,
) -> np.ndarray:
    """
    MBAR dimensionless free energies f_k (f_0 = 0).

    Seeded by :func:`_initial_free_energies` (pymbar's default zero
    initialisation does not converge on REST2 ladders whose adjacent-state
    gaps are hundreds of kT).  pymbar is used when importable; otherwise, or
    if it fails, a plain self-consistent iteration is run.
    """
    u_kn = np.asarray(u_kn, dtype=np.float64)
    N_k = np.asarray(N_k, dtype=np.int64)
    logN = np.log(N_k)
    f = _initial_free_energies(u_kn, N_k) if f_init is None else np.array(f_init, float)

    def _residual(f):
        logD = logsumexp((f - u_kn.T) + logN, axis=1)
        f_new = -logsumexp(-u_kn - logD[None, :], axis=1)
        f_new -= f_new[0]
        return f_new, float(np.max(np.abs(f_new - f)))

    if use_pymbar:
        try:
            import pymbar

            mb = pymbar.MBAR(u_kn, N_k, initial_f_k=f, verbose=False)
            f_p = np.asarray(mb.f_k, dtype=float)
            f_p = f_p - f_p[0]
            _, res = _residual(f_p)
            if np.all(np.isfinite(f_p)) and res < 1e-3:
                return f_p
        except Exception:  # noqa: BLE001 — pymbar is optional
            pass
    for _ in range(max_iter):
        f_new, delta = _residual(f)
        f = f_new
        if delta < tol:
            break
    return f


def mbar_log_weights(
    u_target: np.ndarray, u_kn: np.ndarray, f_k: np.ndarray, N_k: np.ndarray
) -> np.ndarray:
    """
    log w_n for the target reduced potential u_target (shape (N,)), normalised
    so that sum_n w_n = 1.
    """
    logD = logsumexp((f_k - u_kn.T) + np.log(N_k), axis=1)   # (N,)
    logw = -u_target - logD
    return logw - logsumexp(logw)


def effective_sample_size(logw: np.ndarray) -> float:
    return float(np.exp(-logsumexp(2.0 * logw)))


# --------------------------------------------------------------------------- #
# Response functions                                                           #
# --------------------------------------------------------------------------- #


def _weighted_moments(store: OccupancyStore, w: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                      frame_mask: np.ndarray | None = None):
    """
    Weighted first/second moments needed for R1, R2, chunked per state so no
    (N, C) float temporary is materialised.

    w, g1, g2 : (N,) in pooled order (state-major).  frame_mask: (N,) bool.
    Returns dict with m_phi, m_g1, m_g2, s_phi_g1, s_phi_g2, s_phi_g1sq, s_g1sq.
    """
    K, n_fr, C = store.phi.shape
    if frame_mask is not None:
        w = np.where(frame_mask, w, 0.0)
    w = w / w.sum()
    m_phi = np.zeros(C); s_phi_g1 = np.zeros(C); s_phi_g2 = np.zeros(C); s_phi_g1sq = np.zeros(C)
    m_g1 = float(w @ g1); m_g2 = float(w @ g2)
    dg1 = g1 - m_g1
    s_g1sq = float(w @ dg1 ** 2)
    for s in range(K):
        sl = slice(s * n_fr, (s + 1) * n_fr)
        ws = w[sl]
        if not np.any(ws):
            continue
        phi = store.phi[s].astype(np.float32)
        m_phi += ws @ phi
        s_phi_g1 += (ws * dg1[sl]) @ phi
        s_phi_g2 += (ws * (g2[sl] - m_g2)) @ phi
        s_phi_g1sq += (ws * dg1[sl] ** 2) @ phi
    return dict(m_phi=m_phi, m_g1=m_g1, m_g2=m_g2, s_phi_g1=s_phi_g1,
                s_phi_g2=s_phi_g2, s_phi_g1sq=s_phi_g1sq, s_g1sq=s_g1sq)


def _response_from_moments(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(phi_mean, R1, R2) from weighted moments."""
    phi_mean = m["m_phi"]
    # Cov_w(phi, g1) = sum w (g1 - <g1>) phi
    R1 = -m["s_phi_g1"]
    # <dphi (dg1)^2>_c = sum w (g1-<g1>)^2 phi - <phi> Var(g1)
    third = m["s_phi_g1sq"] - phi_mean * m["s_g1sq"]
    R2 = third - m["s_phi_g2"]
    return phi_mean, R1, R2


@dataclass
class LambdaResponse:
    lam: float
    phi_mean: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    ess: float
    R1_err: np.ndarray | None = None
    R2_err: np.ndarray | None = None


class ResponseEstimator:
    """
    Pooled (MBAR) response estimator for one :class:`OccupancyStore`.

    Parameters
    ----------
    store : OccupancyStore
    """

    def __init__(self, store: OccupancyStore, use_pymbar: bool = True):
        self.store = store
        self.u_kn = store.pooled_u_kn()
        self.N_k = np.full(store.n_states, store.n_frames)
        self.a, self.b, self.c, self.fit_residual = store.energy_decomposition()
        self.f_k = mbar_free_energies(self.u_kn, self.N_k, use_pymbar=use_pymbar)
        self._logD = logsumexp((self.f_k - self.u_kn.T) + np.log(self.N_k), axis=1)

    # -- generators at lambda --------------------------------------------------
    def u_at(self, lam: float) -> np.ndarray:
        return self.a + self.b * lam + self.c * np.sqrt(lam)

    def generators(self, lam: float) -> tuple[np.ndarray, np.ndarray]:
        s = np.sqrt(lam)
        g1 = self.b + self.c / (2.0 * s)
        g2 = -self.c / (4.0 * lam * s)
        return g1, g2

    def log_weights(self, lam: float) -> np.ndarray:
        logw = -self.u_at(lam) - self._logD
        return logw - logsumexp(logw)

    def ess(self, lam: float) -> float:
        return effective_sample_size(self.log_weights(lam))

    # -- estimates -----------------------------------------------------------
    def at(self, lam: float, frame_mask: np.ndarray | None = None) -> LambdaResponse:
        logw = self.log_weights(lam)
        w = np.exp(logw)
        g1, g2 = self.generators(lam)
        m = _weighted_moments(self.store, w, g1, g2, frame_mask)
        phi_mean, R1, R2 = _response_from_moments(m)
        return LambdaResponse(lam=float(lam), phi_mean=phi_mean, R1=R1, R2=R2,
                              ess=effective_sample_size(logw))

    def curve(self, lambdas: Sequence[float]) -> list[LambdaResponse]:
        return [self.at(l) for l in lambdas]

    def bootstrap(self, lam: float, n_boot: int = 50, block: int = 50,
                  seed: int = 0) -> LambdaResponse:
        """
        Block bootstrap over frames (blocks of ``block`` consecutive frames
        within each state, resampled with replacement) with f_k held fixed.
        Returns the point estimate with R1_err / R2_err set to bootstrap
        standard deviations.
        """
        rng = np.random.default_rng(seed)
        K, n_fr, _ = self.store.phi.shape
        logw = self.log_weights(lam)
        w = np.exp(logw)
        g1, g2 = self.generators(lam)
        base = self.at(lam)
        n_blocks = int(np.ceil(n_fr / block))
        R1s, R2s = [], []
        for _ in range(n_boot):
            # multiplicity of each block per state -> per-frame multiplicity
            mult = np.empty(K * n_fr)
            for s in range(K):
                counts = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks)
                per_frame = np.repeat(counts, block)[:n_fr]
                mult[s * n_fr:(s + 1) * n_fr] = per_frame
            m = _weighted_moments(self.store, w * mult, g1, g2)
            _, R1, R2 = _response_from_moments(m)
            R1s.append(R1); R2s.append(R2)
        base.R1_err = np.std(R1s, axis=0, ddof=1)
        base.R2_err = np.std(R2s, axis=0, ddof=1)
        return base

    # -- ladder-based comparisons ---------------------------------------------
    def single_state(self, state_idx: int) -> LambdaResponse:
        """
        The single-state cumulant estimate at a sampled state (only that
        state's frames, uniform weights) using the same analytic generators —
        for comparison with the pooled estimate.
        """
        K, n_fr, _ = self.store.phi.shape
        mask = np.zeros(K * n_fr, dtype=bool)
        mask[state_idx * n_fr:(state_idx + 1) * n_fr] = True
        lam = float(self.store.lambdas[state_idx])
        w = mask.astype(float)
        g1, g2 = self.generators(lam)
        m = _weighted_moments(self.store, w, g1, g2)
        phi_mean, R1, R2 = _response_from_moments(m)
        return LambdaResponse(lam=lam, phi_mean=phi_mean, R1=R1, R2=R2, ess=float(n_fr))

    def ladder_slope(self, state_idx: int) -> np.ndarray:
        """Central finite-difference d<phi>/dlambda from the raw per-state frequencies."""
        f = self.store.freqs().to_numpy()
        lam = self.store.lambdas
        k = state_idx
        if k == 0:
            return (f[1] - f[0]) / (lam[1] - lam[0])
        if k == len(lam) - 1:
            return (f[-1] - f[-2]) / (lam[-1] - lam[-2])
        return (f[k + 1] - f[k - 1]) / (lam[k + 1] - lam[k - 1])


# --------------------------------------------------------------------------- #
# Geometry helpers                                                             #
# --------------------------------------------------------------------------- #


def cos_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(abs(a @ b) / (na * nb))


def pc_frame(vec: np.ndarray, loadings: np.ndarray, n_pcs: int = 3) -> dict:
    """
    Coordinates of ``vec`` in the frame of the first ``n_pcs`` PC loading
    vectors (columns of ``loadings``, aligned to ``vec``'s contact order).
    Returns |cos| with each PC and the fraction of |vec|^2 inside their span.
    """
    L = np.asarray(loadings)[:, :n_pcs]
    Q, _ = np.linalg.qr(L)
    proj = Q @ (Q.T @ vec)
    out = {f"cos_pc{i + 1}": cos_abs(vec, L[:, i]) for i in range(L.shape[1])}
    out[f"frac_in_pc1_{L.shape[1]}"] = float((proj @ proj) / (vec @ vec)) if vec @ vec > 0 else 0.0
    return out
